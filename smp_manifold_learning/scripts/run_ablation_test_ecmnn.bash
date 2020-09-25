#!/bin/bash

declare -A manifold
manifold=([1]="3Dsphere" [2]="3Dcircle" [3]="3dof_v2_traj" [4]="6dof_traj")

declare -A expmode
expmode=([0]="normal" [1]="wo_augmentation" [2]="wo_rand_combination_normaleigvecs" [3]="wo_siamese_losses" [4]="wo_nspace_alignment" [5]="noisy_normal" [6]="no_siam_reflection" [7]="no_siam_frac_aug" [8]="no_siam_same_levelvec")

for manifold_id in "${!manifold[@]}"; do
  for expmode_id in 0 1 3 4 5 6 7 8; do  #"${!expmode[@]}"; do
    if [[ $expmode_id -eq 0 ]]; then
      bu=1
      bc=1
      bs=1
      ba=1
      bn=0
      bm="all"
    elif [[ $expmode_id -eq 1 ]]; then
      bu=0
      bc=1
      bs=1
      ba=0
      bn=0
      bm="all"
    elif [[ $expmode_id -eq 2 ]]; then
      bu=1
      bc=0
      bs=1
      ba=1
      bn=0
      bm="all"
    elif [[ $expmode_id -eq 3 ]]; then
      bu=1
      bc=1
      bs=0
      ba=1
      bn=0
      bm="all"
    elif [[ $expmode_id -eq 4 ]]; then
      bu=1
      bc=1
      bs=1
      ba=0
      bn=0
      bm="all"
    elif [[ $expmode_id -eq 5 ]]; then
      bu=1
      bc=1
      bs=1
      ba=1
      bn=1  # noisy data
      bm="all"
    elif [[ $expmode_id -eq 6 ]]; then
      bu=1
      bc=1
      bs=1
      ba=1
      bn=0
      bm=${expmode[$expmode_id]}
    elif [[ $expmode_id -eq 7 ]]; then
      bu=1
      bc=1
      bs=1
      ba=1
      bn=0
      bm=${expmode[$expmode_id]}
    elif [[ $expmode_id -eq 8 ]]; then
      bu=1
      bc=1
      bs=1
      ba=1
      bn=0
      bm=${expmode[$expmode_id]}
    fi
    for randseed in {0..2}; do
      aug_dataloader_save_dir="$(printf "../plot/ecmnn/${manifold[$manifold_id]}/${expmode[$expmode_id]}/")"
      save_dir="$(printf "${aug_dataloader_save_dir}/r%02d/" $randseed)"
      printf "Logging in directory %s\n" $save_dir
      python create_dir_if_not_exist.py -d $save_dir
      log_file="$(printf "%s/log.txt" $save_dir)"
      printf "Log file: %s\n" $log_file
      python train_ecmnn.py -d $manifold_id -u $bu -n $bn -m $bm -c $bc -s $bs -a $ba -r $randseed -p $save_dir -v $aug_dataloader_save_dir -l 1 &> $log_file
    done
  done
done
