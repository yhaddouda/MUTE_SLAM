#!/usr/bin/env python3
"""
auto_run_muteslam_4setups_nomesh.py
-----------------------------------
Batch-run MUTE-SLAM experiments across multiple Replica scenes for FOUR setups.
Sweeping of hash table sizes is DISABLED.
Final Mesh Extraction is DISABLED (requires modification to Mapper.py).

Setups:
  A) CoherentPrime hash fn, morton_sort = False
  B) Morton hash fn,       morton_sort = False
  C) CoherentPrime hash fn, morton_sort = True
  D) Morton hash fn,        morton_sort = True

Usage:
  python3 auto_run_muteslam_4setups_nomesh.py \
      --configs-root configs/Replica \
      --tag fp16 \
      --global-log output/Replica/all_scenes/muteslam_nomesh.log
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: PyYAML is required. Install it with:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# Exclude only office2 by default
DEFAULT_EXCLUDES = {"office2"}


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def find_tegrastats() -> str | None:
    candidates = ["/usr/bin/tegrastats", "/bin/tegrastats", "/usr/sbin/tegrastats", "tegrastats"]
    for c in candidates:
        if shutil.which(c):
            return shutil.which(c)
    return None


def extract_peak_ram_from_file(logfile: Path) -> int | None:
    if not logfile or not Path(logfile).exists():
        return None
    peak = None
    pat = re.compile(r"RAM\s+(\d+)[/ ]")  # captures the 'used' MB
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                val = int(m.group(1))
                if peak is None or val > peak:
                    peak = val
    return peak


def build_exp_name(setup: str, tag: str) -> str:
    """
    Construct experiment folder name based on setup and tag.
    
    setup in {"A", "B", "C", "D"}:
      A: prime + no morton sort      -> noMorton_R128
      B: morton + no morton sort     -> hashfn_morton_R128
      C: prime + morton sort         -> R128
      D: morton + morton sort        -> hashfn_morton_sort_R128
    """
    if setup == "A":
        suffix = "A"
    elif setup == "B":
        suffix = "B"
    elif setup == "C":
        suffix = "C"
    elif setup == "D":
        suffix = "D"
    else:
        raise ValueError(f"Unknown setup: {setup}")
        
    return f"{tag}_{suffix}"


def override_config(base_cfg_path: Path, scene: str, setup: str, tag: str) -> Path:
    """
    Return path to a temp config for the given setup.
    """
    cfg = read_yaml(base_cfg_path)

    # Ensure nodes exist
    cfg.setdefault("data", {})
    cfg.setdefault("encoding", {})
    cfg.setdefault("meshing", {})

    # 1. MUTE-SLAM Encoding Configuration
    enc = cfg["encoding"]
    enc["tcnn"] = True
    enc["type"] = "HashGrid"
    enc["morton_R"] = 128
    # Note: We do NOT set "log2_hashmap_size", allowing default calc.

    # 2. DISABLE MESHING (Requires Mapper.py modification)
    cfg["meshing"]["save_final_mesh"] = False
    # Ensure intermediate meshing is also effectively disabled
    cfg["mapping"]["mesh_freq"] = 100000 

    # 3. Setup Logic
    if setup == "A":
        enc["hash"] = "CoherentPrime"
        enc["morton_sort"] = False
    elif setup == "B":
        enc["hash"] = "Morton"
        enc["morton_sort"] = False
    elif setup == "C":
        enc["hash"] = "CoherentPrime"
        enc["morton_sort"] = True
    elif setup == "D":
        enc["hash"] = "Morton"
        enc["morton_sort"] = True
    else:
        raise ValueError(f"Unknown setup: {setup}")

    # 4. Output Path
    exp_name = build_exp_name(setup, tag)
    base_output = cfg["data"].get("output", f"output/Replica/{scene}")
    if base_output.endswith("/"): 
        base_output = base_output[:-1]
    
    cfg["data"]["output"] = f"{base_output}/{exp_name}"

    # Write temp YAML
    tmp_dir = base_cfg_path.parent
    ensure_parent(tmp_dir)
    tmp_cfg = tmp_dir / f"{scene}_autorun_{setup}.yaml"
    write_yaml(tmp_cfg, cfg)
    return tmp_cfg


def run_one(cfg_path: Path, global_log: Path, scene: str, setup: str, tag: str, python_exec: str) -> None:
    ensure_parent(global_log)

    tegra = find_tegrastats()
    tegra_proc = None
    tegra_tmp = None
    if tegra:
        tegra_tmp = Path(tempfile.mkstemp(prefix="tegrastats_", suffix=".log")[1])
        tegra_cmd = [tegra, "--interval", "1000"]
        tegra_proc = subprocess.Popen(tegra_cmd, stdout=open(tegra_tmp, "w"), stderr=subprocess.DEVNULL)
    else:
        tegra_tmp = None

    start = time.time()
    try:
        # Execute MUTE-SLAM entry point
        cmd = [python_exec, "run.py", str(cfg_path)]
        print(f">>> Running Scene: {scene} | Setup: {setup} | Tag: {tag}")
        subprocess.run(cmd, check=True)
    finally:
        if tegra_proc is not None:
            tegra_proc.terminate()
            try:
                tegra_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                tegra_proc.kill()

    end = time.time()
    runtime = int(round(end - start))

    peak_ram = extract_peak_ram_from_file(tegra_tmp) if tegra_tmp else None
    if tegra_tmp:
        try:
            Path(tegra_tmp).unlink(missing_ok=True)
        except Exception:
            pass

    exp_name = build_exp_name(setup, tag)

    line = {
        "scene": scene,
        "tag": tag,
        "setup": setup,
        "exp_name": exp_name,
        "time_s": runtime,
        "peak_RAM_MB": peak_ram,
    }

    parts = [f"{k}={v}" for k, v in line.items() if v is not None]
    log_line = ", ".join(parts)

    with open(global_log, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

    print(">>> Logged:", log_line)


def discover_scenes(configs_root: Path, excludes: set[str]) -> list[str]:
    """Return list of scene names by scanning configs_root for *.yaml."""
    scenes = []
    if not configs_root.exists():
        return scenes
    for yml in sorted(configs_root.glob("*.yaml")):
        scene = yml.stem
        if scene in ["replica", "SLAM"]:
            continue
        if scene in excludes:
            continue
        scenes.append(scene)
    return scenes


def run_for_scene(scene: str, base_cfg_path: Path, setups: list[str],
                  tag: str, python_exec: str, global_log: Path) -> None:
    for setup in setups:
        tmp_cfg = override_config(
            base_cfg_path=base_cfg_path,
            scene=scene,
            setup=setup,
            tag=tag,
        )
        try:
            run_one(
                cfg_path=tmp_cfg,
                global_log=global_log,
                scene=scene,
                setup=setup,
                tag=tag,
                python_exec=python_exec,
            )
        finally:
            pass


def main():
    parser = argparse.ArgumentParser(description="Batch-run MUTE-SLAM for 4 setups (No Mesh).")
    parser.add_argument("--scene", default=None, help="If provided, run only this scene.")
    parser.add_argument("--scene-cfg", default=None, help="Path to base scene YAML.")
    parser.add_argument("--configs-root", default="configs/Replica", help="Folder containing scene YAMLs.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Explicit list of scenes.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Extra scenes to exclude.")
    parser.add_argument("--tag", default="fp32", help="Short tag for exp_name.")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable.")
    parser.add_argument("--global-log", default="output/Replica/all_scenes/muteslam_nomesh.log", help="Log file.")
    args = parser.parse_args()

    global_log = Path(args.global_log)
    ensure_parent(global_log)

    setups = ["A", "B", "C", "D"]

    # Determine mode (Single vs Multi)
    if args.scene:
        if not args.scene_cfg:
            cfg_path = Path(args.configs_root) / f"{args.scene}.yaml"
        else:
            cfg_path = Path(args.scene_cfg)
        cfg_path = cfg_path.resolve()
        if not cfg_path.exists():
            print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
            sys.exit(2)
        run_for_scene(args.scene, cfg_path, setups, args.tag, args.python_exec, global_log)
        return

    configs_root = Path(args.configs_root).resolve()
    if args.scenes:
        scenes = args.scenes
    else:
        excludes = set(map(str, DEFAULT_EXCLUDES)) | set(map(str, args.exclude))
        scenes = discover_scenes(configs_root, excludes)

    if not scenes:
        print("No scenes selected.", file=sys.stderr)
        sys.exit(0)

    print(f">>> Scenes to run ({len(scenes)}): {', '.join(scenes)}")

    for scene in scenes:
        cfg_path = (configs_root / f"{scene}.yaml").resolve()
        if not cfg_path.exists():
            print(f"WARNING: Skipping {scene} (missing config)", file=sys.stderr)
            continue
        run_for_scene(scene, cfg_path, setups, args.tag, args.python_exec, global_log)


if __name__ == "__main__":
    main()