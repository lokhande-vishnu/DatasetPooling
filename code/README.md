# Equivariance allows Handling Multiple Nuisance Variables when Analyzing Pooled Datasets
We provide code for the experiments done in the paper. The organization of code is as follows :
- All code is present in ``src`` directory. 
- Bash files are provided to run the experiments for the proposed method and the baselines

### Requirements 
We attach the conda `requirements.txt` file to provide 
exact environment details used for running the experiments. 

The main dependencies are 
- pytorch
- tensorflow for tensorboard
- PIL 
- sklearn
- numpy
- torchvision
- MulticoreTSNE
- matplotlib
- tqdm
- nibabel
- seaborn

## Running Experiments
Next, we provide details on how to run each experiment one by one.  
The command to run each baseline with the hyperparameters
used for reporting the results
are present in relevant `run_all_<dataset-name>.sh` file
Most of the code, commands and argument names are self-explanatory.

Running the experiment :
1) Check `run_all_<dataset-name>.sh`. Each baseline is 
run for three random seeds.
To run a baseline comment the appropriate section
of the file.
2) Note that `equiv_lambda` is the hyperparameter 
for regularization weight for this experiment.
3) Run ``bash run_all_<dataset-name>.sh`` with 
the selected baseline and parameters.
4) The results can be seen in `result/<experiment_name>/<run_id>` directory,
where `<run_id>` is a string consisting of all relevant 
parameters used for this experiment run.
5) The stats are available in `result/<experiment_name>/<run_id>/log.txt` file,
compile the stats to generate mean/std evaluations by running,
``python compile_results.py --dataset_name <dataset-name>``

The data required to run this experiment is already provided
in `data/` directory. The train, validation and test 
splits used for reporting results are already
present in this directory. To generate fresh splits
use `src/uci_data.py`.

For ADNI dataset, download the dataset from http://adni.loni.usc.edu/ and create a symbolic link to data using `ln -s <ADNI-data-dir> ./data/adni_data`

## Note on baselines
In every `run_all_<dataset-name>.sh` files, we evaluate four methods
1) `equivar`: This is our proposed method.
2) `none`: This is a baseline without any equivariance/invariance requirements.
3) `zemel`: This is an implementation of (Li, Swersky, and Zemel2014)
4) `cai`: This is an implementation of the adversarial training approach (Xie et al. 2017).
