# Reproducibility study of "Robust Fair Clustering: A Novel Fairness Attack and Defense Framework"

This is the code for the reproduction and extension of the paper [Robust Fair Clustering: A Novel Fairness Attack and Defense Framework](https://arxiv.org/pdf/2210.01953.pdf) by Chhabra et al. The original codebase can be found at [anshuman23/CFC](https://github.com/anshuman23/CFC/tree/master).

## How to run the experiment
### Install requirements
 - Download & Install [IBM CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) on your system. Make sure to run the command to install the related Python module. The exact command to do this will be generated and output by the CPLEX installer.
 - Install the Conda environment with `conda env create -f environment.yml`

### Download the datasets
The following datasets can be acquired and should be saved in `Fair-Clustering-Codebase/fair_clustering/raw_data/`
 - The Office-31, MNIST_USPS are downloaded automatically in the code.
 - The Yale dataset is currently unavailable online, but an unofficial download can be found at [AcademicTorrents](https://academictorrents.com/details/aad8bf8e6ee5d8a3bf46c7ab5adfacdd8ad36247).
 - The FairFace dataset can be downloaded from [dchen236/FairFace](https://github.com/dchen236/FairFace). You only need to download the validation set and the corresponding csv with the labels.
 - The OULAD datset can be downloaded [here](https://analyse.kmi.open.ac.uk/open_dataset). Only the file `studentInfo.csv` is required.
 - The Dutch Census dataset can be downloaded from [tailequy/fairness_dataset](https://github.com/tailequy/fairness_dataset/tree/main/)

### Run the experiments
 You can run the experiments sequentially by clicking _Run All_ in the `reproduction.ipynb` notebook. However this is not recommended as this will take several days to weeks to complete on consumer grade hardware. Instead we recommend to make use of a high-performance compute cluster and use `Fair-Clustering-Codebase/reproduce-attack.py` and `Fair-Clustering-Codebase/reproduce-defense.py`. In order to do this you first have to set some environment variables:
 - ```export OPENBLAS_NUM_THREADS=32``` (set the value to the number of CPU-cores on your system)
 - ```CUBLAS_WORKSPACE_CONFIG=:4096:8``` (`:4096:8` should work on most industrial hardware, set to `:16:8` on consumer grade hardware)

 Now you can run each experiment separately or in parralel. 
  - For the attack: run `python3 Fair-Clustering-Codebase/reproduce-attack.py --algorithm <algorithm> --dataset <dataset> --objective <objective>`
  - For the defense: run `python3 Fair-Clustering-Codebase/reproduce-defense.py --dataset <dataset> --objective <objective>`

In both cases the `<objective>` can be either `balance` or `entropy`, depending on which fairness metric you want to minimize and the `<dataset>` can be substituted with one of `Office-31`, `MNIST_USPS`, `Yale`, `DIGITS`, `OULAD`, `Dutch_Census_2001`, `FairFace`. Note that  `OULAD`, `Dutch_Census_2001` will not work with the defense because these are tabular datasets. In the case of the attack, `<algorithm>` should be substituted with the fair clustering algorithm to attack: either `SFD`, `FSC` or `KFC`.