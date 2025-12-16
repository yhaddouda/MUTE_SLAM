import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate, normalize_3d_coordinate_to_unit

# Morton extension
from src.fast_morton import morton3d_keys_cuda

@torch.no_grad()
def morton_permutation_from_points01(pts01: torch.Tensor, R: int = 128) -> torch.Tensor:
    """
    pts01: [N, 3], in [0,1]^3 on CUDA
    returns: permutation indices [N] as torch.long on CUDA
    """
    with torch.cuda.nvtx.range("morton_keys"):
        keys = morton3d_keys_cuda(pts01, R)  # int32
    with torch.cuda.nvtx.range("morton_argsort"):
        perm = torch.argsort(keys, stable=False)
    return perm.to(torch.long)

def _cast_to_mlp_dtype(x, mlp_layers: nn.ModuleList):
        # pick dtype from the first Linear weight
        if len(mlp_layers) == 0:
            return x
        target_dtype = mlp_layers[0].weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        return x

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta
    """
    def __init__(self, device, in_dim=32, hidden_size=32, truncation=0.08, n_blocks=2, learnable_beta=True, use_tcnn=False, morton_sort=True, morton_R=128):
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.truncation = truncation
        self.n_blocks = n_blocks
        self.bound = torch.empty(3, 2)
        self.use_tcnn = use_tcnn

        ## Morton params
        self.morton_sort = morton_sort
        self.morton_R = morton_R

        ## layers for SDF decoder
        self.linears = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        ## layers for RGB decoder
        self.c_linears = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

        self.output_linear = nn.Linear(hidden_size, 1)
        self.c_output_linear = nn.Linear(hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """

        xy = planes_xy(p_nor[..., [0, 1]])
        xz = planes_xz(p_nor[..., [0, 2]])
        yz = planes_yz(p_nor[..., [1, 2]])
        feat = xy + xz + yz  # [N, 32]

        return feat

    def get_feature_from_points(self, pts, submap_list):
        """
        Get features from inputting points
        Args:
            pts (tensor, (N,3)): normalized 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            features (tensor)
        """
        index = torch.linspace(0, pts.shape[0]-1, pts.shape[0], dtype=torch.long, device=self.device)
        feat_list = []
        c_feat_list = []
        indices_list = []
        pre_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
        for submap in submap_list:
            pts_mask = torch.logical_and((pts[..., :] > submap.boundary[0]).all(dim=-1),
                                         (pts[..., :] < submap.boundary[1]).all(dim=-1))
            pts_mask = torch.logical_and(pts_mask, torch.logical_xor(pre_mask, pts_mask))
            #pre_mask = torch.logical_or(pre_mask, pts_mask)
            pre_mask = pts_mask

            # Indices in the flattened pts array
            indices = index[pts_mask]
            indices_list.append(indices)
            pts_sub = pts[pts_mask]   # [N_i, 3]

            with torch.cuda.nvtx.range("normalize coords"):
                if self.use_tcnn:
                    p_nor = normalize_3d_coordinate_to_unit(pts_sub, submap.boundary)
                else:
                    p_nor = normalize_3d_coordinate(pts_sub, submap.boundary)

            # --- Morton sort only in TCNN mode ---
            if self.use_tcnn and self.morton_sort and p_nor.shape[0] > 0:
                with torch.cuda.nvtx.range("morton_permutation"):
                    # p_nor is already in [0,1]^3 in the TCNN path
                    perm = morton_permutation_from_points01(p_nor, R=self.morton_R)

                p_nor_sorted = p_nor[perm]

                # Query encoders on sorted coordinates
                with torch.cuda.nvtx.range("query color feature"):
                    c_feat_sorted = self.sample_plane_feature(
                        p_nor_sorted,
                        submap.c_planes_xy, submap.c_planes_xz, submap.c_planes_yz
                    )
                with torch.cuda.nvtx.range("query geometry feature"):
                    feat_sorted = self.sample_plane_feature(
                        p_nor_sorted,
                        submap.planes_xy, submap.planes_xz, submap.planes_yz
                    )
                with torch.cuda.nvtx.range("Unpermute"):
                    # Un-permute so that features are back in the original order
                    c_feat = torch.empty_like(c_feat_sorted)
                    c_feat[perm] = c_feat_sorted

                    feat = torch.empty_like(feat_sorted)
                    feat[perm] = feat_sorted

            else:
                # Original behavior (no Morton or non-TCNN)
                with torch.cuda.nvtx.range("query color feature"):
                    c_feat = self.sample_plane_feature(
                        p_nor,
                        submap.c_planes_xy, submap.c_planes_xz, submap.c_planes_yz
                    )
                with torch.cuda.nvtx.range("query geometry feature"):
                    feat = self.sample_plane_feature(
                        p_nor,
                        submap.planes_xy, submap.planes_xz, submap.planes_yz
                    )

            c_feat_list.append(c_feat)
            feat_list.append(feat)

        # allocate outputs with the same dtype as the produced features (fp16 or fp32)
        feat_dtype = feat_list[0].dtype
        c_feat_dtype = c_feat_list[0].dtype

        feat_all = torch.zeros((pts.shape[0], feat_list[0].shape[1]), device=self.device, dtype=feat_dtype)
        c_feat_all = torch.zeros((pts.shape[0], c_feat_list[0].shape[1]), device=self.device, dtype=c_feat_dtype)

        for feat, c_feat, indices in zip(feat_list, c_feat_list, indices_list):
            # index_put_ requires exact dtype match
            if feat.dtype != feat_all.dtype:
                feat = feat.to(feat_all.dtype)
            if c_feat.dtype != c_feat_all.dtype:
                c_feat = c_feat.to(c_feat_all.dtype)

            feat_all.index_put_((indices,), feat)
            c_feat_all.index_put_((indices,), c_feat)

        return feat_all, c_feat_all


    def get_feature_from_points_for_mesher(self, pts, submap_list):
        """
        Get features from inputting points
        Args:
            pts (tensor, (N,3)): normalized 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            features (tensor)
        """
        with torch.no_grad():
            index = torch.linspace(0, pts.shape[0]-1, pts.shape[0], dtype=torch.long, device=self.device)
            feat_list = []
            c_feat_list = []
            indices_list = []
            pre_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
            for submap in submap_list:
                pts_mask = torch.logical_and((pts[..., :] > submap.boundary[0]).all(dim=-1),
                                            (pts[..., :] < submap.boundary[1]).all(dim=-1))
                pts_mask = torch.logical_and(pts_mask, torch.logical_xor(pre_mask, pts_mask))
                #pre_mask = torch.logical_or(pre_mask, pts_mask)
                pre_mask = pts_mask
                indices_list.append(index[pts_mask])
                if self.use_tcnn:
                    p_nor = normalize_3d_coordinate_to_unit(pts[pts_mask], submap.boundary)
                else:
                    p_nor = normalize_3d_coordinate(pts[pts_mask], submap.boundary)
                feat_list.append(self.sample_plane_feature(p_nor, submap.planes_xy, submap.planes_xz, submap.planes_yz))
                c_feat_list.append(self.sample_plane_feature(p_nor, submap.c_planes_xy, submap.c_planes_xz, submap.c_planes_yz))
            feat_all = torch.full((pts.shape[0], feat_list[0].shape[1]), float('nan'), device=self.device)
            c_feat_all = torch.zeros((pts.shape[0], c_feat_list[0].shape[1]), device=self.device)
            for feat, c_feat, indices in zip(feat_list, c_feat_list, indices_list):
                feat_all.index_put_((indices,), feat)
                c_feat_all.index_put_((indices,), c_feat)

            out_bound_indices = index[torch.isnan(feat_all).any(dim=-1)]
        return feat_all, c_feat_all, out_bound_indices

    def get_raw_sdf(self, feat):
        """
        Get raw SDF
        Args:
            feat (tensor, (N, feature dimension))
        Returns:
            sdf (tensor): raw SDF
        """
        h = feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf

    def get_raw_rgb(self, feat):
        """
        Get raw RGB
        Args:
            feat (tensor, (N, feature dimension))
        Returns:
            rgb (tensor): raw RGB
        """
        h = feat
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb

    def get_raw_for_mesher(self, p, submap_list):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape
        p = p.reshape(-1, 3)
        with torch.no_grad():
            features, c_features, out_bound_indices = self.get_feature_from_points_for_mesher(p, submap_list)

            sdf = self.get_raw_sdf(features).detach()
            sdf.index_put_((out_bound_indices,), torch.tensor([1.], device=self.device))
            rgb = self.get_raw_rgb(c_features).detach()

            raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
            raw = raw.reshape(*p_shape[:-1], -1)

        return raw


    def forward(self, p, submap_list):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            submap_list (List): The list of sub_map objects
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape
        p = p.reshape(-1, 3)
        with torch.cuda.nvtx.range("get_features_from_points"):
            features, c_features = self.get_feature_from_points(p, submap_list)

        with torch.cuda.nvtx.range("sdf decoder"):
            features = _cast_to_mlp_dtype(features, self.linears)
            sdf = self.get_raw_sdf(features)
        with torch.cuda.nvtx.range("rgb decoder"):
            c_features = _cast_to_mlp_dtype(c_features, self.c_linears)
            rgb = self.get_raw_rgb(c_features)

        with torch.cuda.nvtx.range("tensor ops"):
            raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)
            raw = raw.reshape(*p_shape[:-1], -1)

        return raw
