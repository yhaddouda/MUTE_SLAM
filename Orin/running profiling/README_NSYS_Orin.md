
# Annotated code with nvtx, then the following command : 
sudo nsys profile   --trace cuda,osrt,nvtx,cudnn,cublas   --gpu-metrics-devices all   --cuda-memory-usage true  --duration 130 --force-overwrite true --output profile_cfgA   /bin/b
ash /home/yh279050/work/MUTE_SLAM/Orin/running\ profiling/profile_muteslam.sh  configs/Replica/office0.yaml

--> profile_muteslam.sh  : important for running cuda morton optimised code, because it tells nsight where to look for the libs and sets the envs variables in conda and not sudo, this requires that the cache already exists in the project folder after a first build

# After generating the .nsys-rep file with the previous command, you can generate statistics for a nvtx range (function) with this command :
nsys stats --report nvtx_sum --format csv --output Orin_Morton_R128_T13.csv Orin_Morton_R128_T13.nsys-rep

