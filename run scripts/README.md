# Universal Batch Runner for MUTE-SLAM

`auto_run_universal.py` is a automation script for batch-running MUTE-SLAM experiments. It allows sweeping across Scenes, Setups, Precision, and Hash Table Sizes.

## Usage

Place this script in the root directory (alongside `run.py`) and run:

```bash
python auto_run_universal.py [FLAGS]

## Flags

Flag,Description,Default
--scenes,Space-separated list of scenes (e.g. office0 room1).,All scenes in configs/Replica
--setups,"Setups to run (A, B, C, D).",A B C D
--precisions,fp16 or fp32.,fp16
--sizes,"Log2 hashmap sizes (e.g. 19, 20). Use null for default.",null
--configs-root,Path to base config folder.,configs/Replica
--log-file,Output CSV file for timings.,output/execution_times.csv

## Example
Run Setup A with fp16 on office0:
python auto_run_universal.py --scenes office0 --setups A --precisions fp16 --sizes null