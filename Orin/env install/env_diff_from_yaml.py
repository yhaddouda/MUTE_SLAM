#!/usr/bin/env python3
# env_diff_from_yaml.py
# A minimal tool to diff a conda YAML environment file against an existing conda env
# and output two lists: missing conda packages and missing pip packages.
#
# Usage:
#   python env_diff_from_yaml.py <environment.yaml> <target_conda_env> [--out-prefix NAME] [--profile PROFILE]
#
# Example:
#   python env_diff_from_yaml.py environment_mute.yaml coslam --out-prefix ismap --profile jetson
#
# Notes:
# - This parser doesn't need PyYAML; it uses a simple state machine and should handle typical env files.
# - It compares names (ignoring versions/build strings) to answer: "what package names from YAML are not present in my env?"
# - The 'jetson' profile removes packages that are known-problematic on Jetson (MKL, open3d, qt/pyqt, embree, pytorch-mutex, pip openexr).

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def parse_env_yaml(path):
    # Very small parser tailored for conda 'environment.yaml' common structure.
    # Returns (conda_names, pip_names).
    # - conda_names: set of package names (lowercase) from top-level dependencies
    # - pip_names: set of package names (lowercase) from the 'pip:' sublist
    # Names are normalized: lowercase, '-' and '_' treated the same; version/build info stripped.
    text = read_text(path)
    lines = text.splitlines()

    def norm_name(name):
        # Strip channels pinning like "pytorch::pytorch"
        if "::" in name:
            name = name.split("::", 1)[1]
        # drop version/build: e.g., 'pytorch3d=0.7.1=py37_cu113_pyt1110' -> 'pytorch3d'
        name = name.strip()
        if name.startswith("- "):
            name = name[2:]
        name = name.strip()
        if not name or name == "pip":
            return None
        name = name.split("=", 1)[0]
        name = name.strip()
        # final normalization
        return name.lower().replace("_","-")

    conda_names = set()
    pip_names = set()

    in_deps = False
    in_pip = False
    for raw in lines:
        line = raw.rstrip("\n")
        if line.strip().startswith("dependencies:"):
            in_deps = True
            in_pip = False
            continue
        if not in_deps:
            continue

        # Detect entering pip subsection
        if re.match(r"^\s*-\s*pip\s*:\s*$", line):
            in_pip = True
            continue

        if in_pip:
            # pip entries look like "    - package==version"
            m = re.match(r"^\s{2,}-\s+(.+?)\s*$", line)
            if m:
                spec = m.group(1).strip()
                # Keep name only before '==', '>=', '@', etc.
                name = re.split(r"\s*(==|>=|<=|~=|!=|>|<|@)\s*", spec, maxsplit=1)[0]
                name = name.strip().lower().replace("_","-")
                if name:
                    pip_names.add(name)
            # Leaving pip subsection if indentation resets or another top-level dash appears
            if re.match(r"^\S", line):
                in_pip = False
            continue

        # top-level conda deps entries: "  - name[=ver[=build]]" or channels like "pytorch::pytorch"
        m = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if m:
            name = norm_name(m.group(1))
            if name:
                conda_names.add(name)

        # crude end condition if another section begins
        if re.match(r"^\S", line) and not line.strip().startswith("- "):
            pass

    # Remove accidental 'pip' in conda_names if present
    conda_names.discard("pip")
    return conda_names, pip_names

def get_installed_conda(env_name):
    # Return a set of normalized conda package names installed in the env.
    try:
        out = subprocess.check_output(["conda", "list", "-n", env_name, "--json"], stderr=subprocess.STDOUT)
        data = json.loads(out.decode())
        names = set()
        for rec in data:
            name = rec.get("name","").strip().lower().replace("_","-")
            if name:
                names.add(name)
        return names
    except Exception as e:
        print(f"[WARN] conda list --json failed: {e}\nFalling back to 'conda list -n {env_name}' parsing...", file=sys.stderr)
        try:
            out = subprocess.check_output(["conda", "list", "-n", env_name], stderr=subprocess.STDOUT).decode()
            names = set()
            for line in out.splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts:
                    names.add(parts[0].strip().lower().replace("_","-"))
            return names
        except Exception as e2:
            print(f"[ERROR] Failed to obtain conda packages from env '{env_name}': {e2}", file=sys.stderr)
            return set()

def get_installed_pip(env_name):
    # Return a set of normalized pip package names installed in the env.
    try:
        out = subprocess.check_output(["conda", "run", "-n", env_name, "python", "-m", "pip", "freeze"], stderr=subprocess.STDOUT)
        names = set()
        for line in out.decode().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            if "@" in line:
                name = line.split("@",1)[0].strip()
            else:
                name = re.split(r"[=<>!~]", line, maxsplit=1)[0].strip()
            name = name.lower().replace("_","-")
            if name:
                names.add(name)
        return names
    except Exception as e:
        print(f"[WARN] pip freeze via conda run failed: {e}", file=sys.stderr)
        return set()

JETSON_FILTERS = {
    # conda names to skip
    "conda": {
        "mkl", "mkl-service", "mkl-fft", "mkl_random", "blas", "libblas", "libcblas", "liblapack",
        "pytorch-mutex", "pyqt", "qt", "embree", "pyembree",
    },
    # pip names to skip or handle separately
    "pip": {
        "torch", "torchvision", "torchaudio",  # install NVIDIA Jetson wheels manually
        "open3d",  # build from source if needed
        "openexr", # usually not needed; prefer OpenEXR libs for OpenCV
    }
}

def apply_profile_filters(missing_conda, missing_pip, profile):
    if profile is None:
        return missing_conda, missing_pip

    p = profile.lower()
    if p == "jetson":
        missing_conda = {n for n in missing_conda if n not in JETSON_FILTERS["conda"]}
        missing_pip = {n for n in missing_pip if n not in JETSON_FILTERS["pip"]}
    return missing_conda, missing_pip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", help="Path to environment.yaml")
    ap.add_argument("env", help="Existing conda env name to compare against (e.g., coslam)")
    ap.add_argument("--out-prefix", default="envdiff", help="Prefix for output files")
    ap.add_argument("--profile", default=None, help="Optional filter profile (e.g., 'jetson')")
    args = ap.parse_args()

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"[ERROR] YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(2)

    desired_conda, desired_pip = parse_env_yaml(yaml_path)
    installed_conda = get_installed_conda(args.env)
    installed_pip = get_installed_pip(args.env)

    # Compute missing
    missing_conda = sorted(desired_conda - installed_conda)
    missing_pip = sorted(desired_pip - installed_pip)

    # Apply optional profile filters
    if args.profile:
        mconda, mpip = apply_profile_filters(set(missing_conda), set(missing_pip), args.profile)
        missing_conda = sorted(mconda)
        missing_pip = sorted(mpip)

    # Write outputs
    out_conda = Path(f"{args.out_prefix}_missing_conda.txt")
    out_pip = Path(f"{args.out_prefix}_missing_pip.txt")
    out_conda.write_text("\n".join(missing_conda) + ("" if missing_conda else ""), encoding="utf-8")
    out_pip.write_text("\n".join(missing_pip) + ("" if missing_pip else ""), encoding="utf-8")

    # Pretty print summary
    print(f"[TARGET ENV] {args.env}")
    print(f"[YAML] {yaml_path.name}")
    print(f"[PROFILE] {args.profile or '(none)'}\n")
    print(f"Missing conda packages ({len(missing_conda)}):")
    for n in missing_conda:
        print("  -", n)
    print(f"\nMissing pip packages ({len(missing_pip)}):")
    for n in missing_pip:
        print("  -", n)
    print(f"\nWrote: {out_conda} and {out_pip}")

if __name__ == "__main__":
    main()
