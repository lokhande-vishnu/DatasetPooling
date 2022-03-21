#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

neps=501
aneps=200

prog_equivar () {
exp='all_german_fairness_equivar'
for seed in 685 648 535
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'German' \
	   --latent_dim 30 \
           --num_epochs "$neps" \
           --lr 1e-2 \
           --adv_lr 1e-2 \
           --recon_lambda 1e-2 \
	   --comp_lambda 1e-2 \
           --equiv_lambda 4.0 \
	   --equiv_type 'ell2' \
	   --adv_num_epochs "$aneps" \
           --seed "$seed" \
	   --alpha 0.1 \
	   --alpha_max 25.0 \
	   --flag_train_equivar \
	   --flag_train_invar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_none () {
exp='all_german_fairness_none'
for seed in 685 648 535
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'German' \
	   --latent_dim 30 \
           --num_epochs "$neps" \
           --lr 1e-2 \
           --adv_lr 1e-2 \
           --recon_lambda 1e-2 \
	   --comp_lambda 1e-2 \
           --equiv_lambda 4.0 \
	   --equiv_type 'none' \
	   --adv_num_epochs "$aneps" \
           --seed "$seed" \
	   --alpha 0.1 \
	   --alpha_max 25.0 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv  
done
}
prog_zemel () {
exp='all_german_fairness_zemel'
for seed in 685 648 535
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'German' \
	   --latent_dim 30 \
           --num_epochs "$neps" \
           --lr 1e-2 \
           --adv_lr 1e-2 \
           --recon_lambda 1e-2 \
	   --comp_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'mmd_lap' \
	   --mmd_lap_p 10.0 \
	   --adv_num_epochs "$aneps" \
           --seed "$seed" \
	   --alpha 0.1 \
	   --alpha_max 25.0 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_cai () {
exp='all_german_fairness_cai'
for seed in 685 648 535
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'German' \
	   --latent_dim 30 \
           --num_epochs "$neps" \
           --lr 1e-2 \
           --adv_lr 1e-2 \
           --recon_lambda 1e-2 \
	   --comp_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'cai' \
	   --mmd_lap_p 10.0 \
	   --adv_num_epochs "$aneps" \
           --seed "$seed" \
	   --alpha 0.1 \
	   --alpha_max 0.1 \
	   --alpha_gamma 1.1 \
	   --alpha_max 25.0 \
	   --disc_lr 1e-2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv
done
}
prog_subsam () {
exp='all_german_fairness_subsam'
for seed in 685 648 535
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'German' \
	   --latent_dim 30 \
           --num_epochs "$neps" \
           --lr 1e-2 \
           --adv_lr 1e-2 \
           --recon_lambda 1e-2 \
	   --comp_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'subsampling' \
	   --mmd_lap_p 10.0 \
	   --adv_num_epochs "$aneps" \
           --seed "$seed" \
	   --alpha 0.1 \
	   --alpha_max 0.1 \
	   --alpha_gamma 1.1 \
	   --alpha_max 25.0 \
	   --disc_lr 1e-2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_randmatch () {
exp='all_german_fairness_randmatch'
for seed in 685 648 535
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'German' \
	   --latent_dim 30 \
           --num_epochs "$neps" \
           --lr 1e-2 \
           --adv_lr 1e-2 \
           --recon_lambda 1e-2 \
	   --comp_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'randmatch' \
	   --adv_num_epochs "$aneps" \
           --seed "$seed" \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_equivar & prog_none & prog_zemel & prog_cai & prog_subsam & prog_randmatch;
