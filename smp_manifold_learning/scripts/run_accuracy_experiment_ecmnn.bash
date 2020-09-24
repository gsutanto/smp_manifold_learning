#!/bin/bash

declare -A manifold
manifold=([1]="3Dsphere" [2]="3Dcircle" [3]="3dof_v2_traj" [4]="6dof_traj")

for randseed in {1..3}; do
  for manifold_id in "${!manifold[@]}"; do
    aug_dataloader_save_dir="$(printf "../plot/ecmnn/${manifold[$manifold_id]}/")"
    save_dir="$(printf "${aug_dataloader_save_dir}/r%02d/" $randseed)"
    printf "Logging in directory %s\n" $save_dir
    python3 create_dir_if_not_exist.py -d $save_dir
    log_file="$(printf "%s/log.txt" $save_dir)"
    printf "Log file: %s\n" $log_file
    python3 train_ecmnn.py -d $manifold_id -r $randseed -p $save_dir -v $aug_dataloader_save_dir -l 1 &> $log_file
  done
done
