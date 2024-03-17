import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import random
import kmedoids
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from zoopt import Dimension, ValueType, Objective, Parameter, Opt, ExpOpt
import seaborn as sns
import time
import argparse
import json

import warnings 
warnings.filterwarnings('ignore')

from fair_clustering.eval.functions import *

from fair_clustering.dataset import ExtendedYaleB, Office31, MNISTUSPS, DutchCensusData, OULADData, FairFace
from fair_clustering.algorithm import FairSpectral, FairKCenter, FairletDecomposition, ScalableFairletDecomposition


def get_fair_clustering_algo(name, cl_algo, n_clusters, random_state):
  if cl_algo == 'FSC':
    if name == 'MNIST_USPS':
      metr_str = 'manhattan'
    else:
      metr_str = 'euclidean'
    fair_clustering_algo = FairSpectral(n_clusters=n_clusters, num_neighbors=3, metric_str=metr_str, random_state=random_state)
  if cl_algo =='SFD':
    if name == 'DIGITS':      
      fair_clustering_algo = ScalableFairletDecomposition(n_clusters=n_clusters, alpha=5, beta=1, random_state=random_state) #5,2
    else:
      fair_clustering_algo = ScalableFairletDecomposition(n_clusters=n_clusters, alpha=5, beta=2, random_state=random_state) #5,2
  # added for KFC with delta paremeter from the paper
  if cl_algo == 'KFC':
    fair_clustering_algo = FairKCenter(n_clusters=n_clusters, delta=0.1, random_state=random_state)

  return fair_clustering_algo

def attack_balance(solution):
  X_copy, s_copy = X.copy(), s.copy()
  flipped_labels = solution.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  fair_clustering_algo = get_fair_clustering_algo(name, cl_algo, n_clusters, random_state) 
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)

  bal = balance(labels_sfd_eval, X_eval, s_eval)

  return bal


def attack_entropy(solution):
  X_copy, s_copy = X.copy(), s.copy()
  flipped_labels = solution.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  fair_clustering_algo = get_fair_clustering_algo(name, cl_algo, n_clusters, random_state) 
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)

  ent = entropy(labels_sfd_eval, s_eval)

  return ent

def process_solution(sol):
  X_copy, s_copy, y_copy = X.copy(), s.copy(), y.copy()
  flipped_labels = sol.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  fair_clustering_algo = get_fair_clustering_algo(name, cl_algo, n_clusters, random_state)
  
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  y_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
    y_eval.append(y_copy[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)
  y_eval = np.array(y_eval)

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  ent = entropy(labels_sfd_eval, s_eval)
  accuracy = acc(y_eval, labels_sfd_eval)
  nmi_score = nmi(y_eval, labels_sfd_eval)

  return (bal, ent, accuracy, nmi_score)


def conduct_random_attack(size_sol):
  X_copy, s_copy, y_copy = X.copy(), s.copy(), y.copy()
  random.seed(None)
  flipped_labels = [random.randint(0,1) for _ in range(size_sol)]
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  fair_clustering_algo = get_fair_clustering_algo(name, cl_algo, n_clusters, random_state)

  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  y_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
    y_eval.append(y_copy[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)
  y_eval = np.array(y_eval)

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  ent = entropy(labels_sfd_eval, s_eval)
  accuracy = acc(y_eval, labels_sfd_eval)
  nmi_score = nmi(y_eval, labels_sfd_eval)

  return (bal, ent, accuracy, nmi_score)



parser = argparse.ArgumentParser(description='Run attack experiments')
parser.add_argument('--dataset', choices=['MNIST_USPS', 'Office-31', 'Yale', 'DIGITS', 'Dutch_Census_2001', 'OULAD', 'FairFace', 'DiabeticData'], type=str, required=True, help='Dataset to run attack experiments on')
parser.add_argument('--algorithm', choices=['SFD', 'FSC', 'KFC'], type=str, required=True, help='Algorithm to run attack experiments on')
parser.add_argument('--objective', choices=['balance', 'entropy'], type=str, required=True, default='balance', help='Objective to run attack experiments on')
args = parser.parse_args()

objective = args.objective
name = args.dataset
cl_algo = args.algorithm

print(f"Dataset: {name}, Algorithm: {cl_algo}, Objective: {objective}")

if name == 'Office-31':
  dataset = Office31(download=False, exclude_domain='amazon', use_feature=True)
  X, y, s = dataset.data
elif name == 'MNIST_USPS':
  dataset = MNISTUSPS(download=False)
  X, y, s = dataset.data
elif name == 'Yale':
  dataset = ExtendedYaleB(download=False, resize=True)
  X, y, s = dataset.data
elif name == 'DIGITS':
  X, y, s = np.load('X_' + name + '.npy'), np.load('y_' + name + '.npy'), np.load('s_' + name + '.npy')
elif name == 'Dutch_Census_2001':
  dataset = DutchCensusData()
  X, y, s = dataset.data
elif name == 'OULAD':
  dataset = OULADData()
  X, y, s = dataset.data
elif name == 'FairFace':
  dataset = FairFace()
  X, y, s = dataset.data


n_clusters = len(np.unique(y))
print("# of clusters -> " + str(n_clusters))
seeds = [150, 1, 4200, 424242, 1947, 355, 256, 7500, 99999, 18]
n_trials = len(seeds)

if name == 'Dutch_Census_2001' or name == 'OULAD' or name == 'FairFace':
  U_idx_full = random.sample(range(len(X)), k=int(len(X) * 0.5))
  V_idx_full = list(set(range(len(X))) - set(U_idx_full))
else:
  U_idx_full, V_idx_full = np.load('U_idx_' + name + '.npy').tolist(), np.load('V_idx_' + name + '.npy').tolist()

pre_attack_res = {
    0 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    1 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    2 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    3 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    4 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    5 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    6 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    7 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
}
post_attack_res = {
    0 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    1 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    2 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    3 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    4 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    5 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    6 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    7 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
}
random_attack_res = {
    0 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    1 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    2 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    3 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    4 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    5 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    6 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
    7 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
}

for percent, j in enumerate([int(0.125*len(U_idx_full)), int(0.25*len(U_idx_full)), int(0.375*len(U_idx_full)), int(0.5*len(U_idx_full)), int(0.625*len(U_idx_full)), int(0.75*len(U_idx_full)), int(0.875*len(U_idx_full)), int(len(U_idx_full))]):
  
  print(f"{percent}/7")

  U_idx = U_idx_full[:j]
  V_idx = V_idx_full

  for trial_idx in range(n_trials):
    random_state = seeds[trial_idx]

    fair_algo = get_fair_clustering_algo(name, cl_algo, n_clusters, random_state)
    fair_algo.fit(X, s)
    labels = fair_algo.labels_

    s_test = []
    X_test = []
    labels_test = []
    y_test = []
    for idx in V_idx:
      s_test.append(s[idx])
      X_test.append(X[idx])
      labels_test.append(labels[idx])
      y_test.append(y[idx])
    s_test = np.array(s_test)
    X_test = np.array(X_test)
    labels_test = np.array(labels_test)
    y_test = np.array(y_test)
    
    pre_attack_res[percent]['BALANCE'].append(balance(labels_test, X_test, s_test))
    pre_attack_res[percent]['ENTROPY'].append(entropy(labels_test, s_test))
    pre_attack_res[percent]['ACC'].append(acc(y_test, labels_test))
    pre_attack_res[percent]['NMI'].append(nmi(y_test, labels_test))
    
    dim_size = len(U_idx)
    dim = Dimension(dim_size, [[0, 1]]*dim_size, [False]*dim_size)

    if objective == 'balance':
      obj = Objective(attack_balance, dim)
    elif objective == 'entropy':
      obj = Objective(attack_entropy, dim)
    
    # added dynamic budget as the authors described in the comment below
    if name == 'MNIST_USPS':
      if cl_algo == 'SFD':
        budget = 50
      elif cl_algo == 'FSC':
        budget = 10
      elif cl_algo == 'KFC':
        budget = 10
    elif name == 'Office-31':
      budget = 20
    elif name == 'Yale':
      if cl_algo == 'SFD':
        budget = 20
      elif cl_algo == 'FSC':
        budget = 10
      elif cl_algo == 'KFC':
        budget = 10
    elif name == 'DIGITS':
      if cl_algo == 'SFD':
        budget = 25
      elif cl_algo == 'FSC':
        budget = 15
      elif cl_algo == 'KFC':
        budget = 15
    elif name == 'Dutch_Census_2001' or name == 'OULAD' or name == "FairFace":
      if cl_algo == 'SFD':
        budget = 30
      elif cl_algo == 'FSC':
        budget = 12
      elif cl_algo == 'KFC':
        budget = 12

    solution = Opt.min(obj, Parameter(budget=budget)) # 10 for FSC for MNIST_USPS and 50 for SFD for MNIST_USPS || 20 for FSC for Office-31 and 20 for SFD for Office-31 || 10 for FSC for Yale and 20 for SFD for Yale || 15 for FSC for DIGITS and 25 for SFD for DIGITS
    
    pa_bal, pa_ent, pa_acc, pa_nmi = process_solution(solution)
    post_attack_res[percent]['BALANCE'].append(pa_bal)
    post_attack_res[percent]['ENTROPY'].append(pa_ent)
    post_attack_res[percent]['ACC'].append(pa_acc)
    post_attack_res[percent]['NMI'].append(pa_nmi)

    r_bal, r_ent, r_acc, r_nmi = conduct_random_attack(dim_size)
    random_attack_res[percent]['BALANCE'].append(r_bal)
    random_attack_res[percent]['ENTROPY'].append(r_ent)
    random_attack_res[percent]['ACC'].append(r_acc)
    random_attack_res[percent]['NMI'].append(r_nmi)

with open('../reproduced-results/attack/' + name + '_' + cl_algo + '_' + objective + '.json', 'a') as f:
  json.dump({'pre_attack_res': pre_attack_res, 'post_attack_res': post_attack_res, 'random_attack_res': random_attack_res}, f)
