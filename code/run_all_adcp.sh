#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

neps=150
aneps=200
prog_equivar () {
exp='all_adcp_harmonz_equivar'
for split in 0 1 2 3 4
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'ADCP' \
           --fold "$split" \
	   --batch_size 32 \
           --num_epochs "$neps" \
           --lr 3e-4 \
           --blocks 2 2 2 2 \
           --channels 16 16 32 64 \
	   --latent_dim 64 \
           --recon_lambda 1e-2 \
           --equiv_lambda 4.0 \
	   --equiv_type 'ell2' \
	   --adv_num_epochs "$aneps" \
	   --adv_lr 1e-3 \
	   --adv_hidden_layers 3 \
	   --adv_hidden_dim 64 \
	   --adv_hidden_layers 3 \
	   --adv_batch_size 16 \
           --alpha 0.1 \
           --alpha_max 1.0 \
           --alpha_gamma 1.2 \
	   --flag_train_equivar \
	   --flag_train_invar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_none () {
exp='all_adcp_harmonz_none'
for split in 0 1 2 3 4
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'ADCP' \
           --fold "$split" \
	   --batch_size 32 \
           --num_epochs "$neps" \
           --lr 3e-4 \
           --blocks 2 2 2 2 \
           --channels 16 16 32 64 \
	   --latent_dim 64 \
           --recon_lambda 1e-2 \
           --equiv_lambda 4.0 \
	   --equiv_type 'none' \
	   --adv_num_epochs "$aneps" \
	   --adv_lr 1e-3 \
	   --adv_hidden_layers 3 \
	   --adv_hidden_dim 64 \
	   --adv_hidden_layers 3 \
	   --adv_batch_size 16 \
           --alpha 0.1 \
           --alpha_max 1.0 \
           --alpha_gamma 1.2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv
done
}
prog_zemel () {
exp='all_adcp_harmonz_zemel'
for split in 0 1 2 3 4
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'ADCP' \
           --fold "$split" \
	   --batch_size 32 \
           --num_epochs "$neps" \
           --lr 3e-4 \
           --blocks 2 2 2 2 \
           --channels 16 16 32 64 \
	   --latent_dim 64 \
           --recon_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'mmd_lap' \
	   --adv_num_epochs "$aneps" \
	   --adv_lr 1e-3 \
	   --adv_hidden_layers 3 \
	   --adv_hidden_dim 64 \
	   --adv_hidden_layers 3 \
	   --adv_batch_size 16 \
           --alpha 0.0001 \
           --alpha_max 0.1 \
           --alpha_gamma 1.2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv
done
}
prog_cai () {
exp='all_adcp_harmonz_cai'
for split in 0 1 2 3 4
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'ADCP' \
           --fold "$split" \
	   --batch_size 32 \
           --num_epochs "$neps" \
           --lr 3e-4 \
           --blocks 2 2 2 2 \
           --channels 16 16 32 64 \
	   --latent_dim 64 \
           --recon_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'cai' \
	   --adv_num_epochs "$aneps" \
	   --adv_lr 1e-3 \
	   --adv_hidden_layers 3 \
	   --adv_hidden_dim 64 \
	   --adv_hidden_layers 3 \
	   --adv_batch_size 16 \
           --alpha 0.0001 \
           --alpha_max 0.1 \
           --alpha_gamma 1.2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv
done
}
prog_subsam () {
exp='all_adcp_harmonz_subsam'
for split in 0 1 2 3 4
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'ADCP' \
           --fold "$split" \
	   --batch_size 32 \
           --num_epochs "$neps" \
           --lr 3e-4 \
           --blocks 2 2 2 2 \
           --channels 16 16 32 64 \
	   --latent_dim 64 \
           --recon_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'subsampling' \
	   --adv_num_epochs "$aneps" \
	   --adv_lr 1e-3 \
	   --adv_hidden_layers 3 \
	   --adv_hidden_dim 64 \
	   --adv_hidden_layers 3 \
	   --adv_batch_size 16 \
           --alpha 0.0001 \
           --alpha_max 0.1 \
           --alpha_gamma 1.2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_randmatch () {
exp='all_adcp_harmonz_randmatch'
for split in 0 1 2 3 4
do
    python src/main.py \
           --experiment_name "$exp" \
           --dataset_name 'ADCP' \
           --fold "$split" \
	   --batch_size 32 \
           --num_epochs "$neps" \
           --lr 3e-4 \
           --blocks 2 2 2 2 \
           --channels 16 16 32 64 \
	   --latent_dim 64 \
           --recon_lambda 1e-2 \
           --equiv_lambda 1e-2 \
	   --equiv_type 'randmatch' \
	   --adv_num_epochs "$aneps" \
	   --adv_lr 1e-3 \
	   --adv_hidden_layers 3 \
	   --adv_hidden_dim 64 \
	   --adv_hidden_layers 3 \
	   --adv_batch_size 16 \
           --alpha 0.0001 \
           --alpha_max 0.1 \
           --alpha_gamma 1.2 \
	   --flag_train_equivar \
	   --flag_test_tsne \
	   --flag_test_mmd \
	   --flag_test_adv 
done
}
prog_equivar & prog_none & prog_zemel & prog_cai;
