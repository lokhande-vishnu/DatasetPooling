'''
Simple script to compile all the results from the log files, once training 
and evaluation are complete.

The mean and standard deviation over three random runs for Adult, German
dataset and over five random splits for ADNI, NIH datasets are computed
'''
import numpy as np
import argparse
import os
import sys
import pdb

sys.path.append('../..')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='German')
parser.add_argument('--result_path', type=str, default='result', help='output path')

args = parser.parse_args()

def main():
    if args.dataset_name == 'German':
        subdirs = ['run_685', 'run_648', 'run_535']
        maindirs = ['all_german_fairness_none', 'all_german_fairness_zemel', 'all_german_fairness_cai', 'all_german_fairness_equivar', 'all_german_fairness_subsam', 'all_german_fairness_randmatch']
    elif args.dataset_name == 'Adult':
        subdirs = ['run_685', 'run_648', 'run_535']
        maindirs = ['all_adult_fairness_none', 'all_adult_fairness_zemel', 'all_adult_fairness_cai', 'all_adult_fairness_equivar', 'all_adult_fairness_subsam', 'all_adult_fairness_randmatch']
    elif args.dataset_name == 'ADNI':
        subdirs = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
        maindirs = ['all_adni_fairness_none', 'all_adni_fairness_zemel', 'all_adni_fairness_cai', 'all_adni_fairness_equivar', 'all_adni_fairness_subsam', 'all_adni_fairness_randmatch']
    elif args.dataset_name == 'NIH':
        subdirs = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
        maindirs = ['all_nih_harmonz_none', 'all_nih_harmonz_zemel', 'all_nih_harmonz_cai', 'all_nih_harmonz_equivar','all_nih_harmonz_subsam', 'all_nih_harmonz_randmatch']
    else:
        raise NotImplementedError
        

    for md in maindirs:
        arr_delta = []
        arr_adv = []
        arr_m = []
        arr_acc =[]
        arr_recons = []
        
        for sd in subdirs:
            log_path = os.path.join(args.result_path, md, sd, "log.txt")

            with open(log_path) as fp:
                line = fp.readline()
                while line:
                    line = line.strip()

                    if 'equivar' in md:
                        if 'INVAR Best' in line:
                            if args.dataset_name == 'German' or args.dataset_name == 'Adult':
                                line = fp.readline()
                                acc = float(line.split('test_acc')[1].split(' ')[0])
                                rec = float(line.split('test_recons')[1].split(' ')[0])
                            else:
                                acc = float(line.split('val_acc')[1].split(' ')[0])
                                rec = float(line.split('val_recons')[1].split(' ')[0])
                    else:
                        if 'Best' in line: 
                            if args.dataset_name == 'German' or args.dataset_name == 'Adult':
                                line = fp.readline()
                                acc = float(line.split('test_acc')[1].split(' ')[0])
                                rec = float(line.split('test_recons')[1].split(' ')[0])
                            else:
                                acc = float(line.split('val_acc')[1].split(' ')[0])
                                rec = float(line.split('val_recons')[1].split(' ')[0])
                                
                    if "test_measure_gap" in line:
                        arr_delta.append(float(line.split('test_measure_gap')[-1]))
                    elif "mmd_measure_loss" in line:
                        arr_m.append(float(line.split('mmd_measure_loss')[-1]))
                    elif "ADVERSARIAL test_acc" in line:
                        if args.dataset_name == 'German':
                            arr_adv.append(float(line.split('roc_auc')[-1]))
                        else:
                            arr_adv.append(float(line.split('ADVERSARIAL test_acc')[1].split(' ')[0]))
                    line = fp.readline()

            # save the acc for every directory
            arr_acc.append(acc)
            arr_recons.append(rec)
        # compute mean and sd
        print(md)
        print('ARR Delta Eq', arr_delta)
        print('ARR Adv', arr_adv)
        print('ARR MMD', arr_m)
        print('ARR ACC', arr_acc)
        print('ARR recons', arr_recons)
        
        print('Delta Eq', np.mean(arr_delta), np.std(arr_delta))
        print('Adv', np.mean(arr_adv), np.std(arr_adv))
        print('MMD', np.mean(arr_m), np.std(arr_m))
        print('ACC', np.mean(arr_acc), np.std(arr_acc))
        print('recons', np.mean(arr_recons), np.std(arr_recons))
        print(' ')
        
if __name__ == '__main__':
    main()
