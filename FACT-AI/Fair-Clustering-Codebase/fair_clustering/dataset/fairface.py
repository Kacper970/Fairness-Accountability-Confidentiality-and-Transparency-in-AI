import os
import sys
import numpy as np
from PIL import Image
import pandas as pd

from fair_clustering.dataset import ImageDataset


class FairFace(ImageDataset):
    """ https://github.com/dchen236/FairFace """

    dataset_name = "FairFace"
    dataset_dir = "fair_clustering/raw_data/fairface"

    def __init__(self, center=True):
        dataset_dir = "fair_clustering/raw_data/fairface"

        labels = pd.read_csv(os.path.join(dataset_dir, '0_labels.csv'))
        labels = labels[labels['race'].isin(['Black', 'White'])]
        labels['race'] = labels['race'].map({'Black': 0, 'White': 1})
        labels['gender'] = labels["gender"].map({'Male': 0, 'Female': 1})
        y = labels["race"].to_numpy()
        s = labels['gender'].to_numpy()

        all_img_paths = [os.path.join(dataset_dir, file[4:]) for file in labels['file'].tolist()]

        X = [np.array(Image.open(p)) for p in all_img_paths]
        X = np.asarray(X)
        X = np.reshape(X, (X.shape[0], -1))

        print(X.shape, y.shape, s.shape)        

        super(FairFace, self).__init__(
            X=X,
            y=y,
            s=s,
            center=center,
        )


if __name__ == "__main__":
    dataset = FairFace()
    X, y, s = dataset.data
    stat = dataset.stat
