# Changes
This file keeps track of all changes made to the original base code in order to make it runnable. With the latest versions of all modules (as the required versions were not specified in the original repository). This file only includes changes to the base code and not the notebooks.

## Changes made
 - Added pandas, seaborn and networkx to requirements.
 - Removed the path prefix from `dataset_dir` in `Fair-Clustering-Codebase/fair_clustering/dataset/mnist_usps.py`, `extended_yaleB.py` and `office31.py` line 13.
 - Changed `np.int` to `np.int8` in `Fair-Clustering-Codebase/fair_clustering/dataset/base.py` lines 75, 76.
 - The provided url for the Yale dataset was dead. We found a mirror which did not support downloading with python. We downloaded the dataset manually and updated `extended_yaleB.py` by commenting out the downloading line.
 - In `fair_kcenter.py` line 191 there is a try-except block which essentially suppressess the error when CPLEX is not installed, causing the KFC algorithm to generate to loop forever without being clear why. Espesially as it is nowhere mentioned that CPLEX is required. We changed it such that it prints the exception instead of suppressing it.
 - Changed `fair_spectral.py` to use a GPU-accelerated version of k-means.