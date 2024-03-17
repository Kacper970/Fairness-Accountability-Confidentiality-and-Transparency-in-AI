import os
import numpy as np
from scipy.io import arff
import pandas as pd
from fair_clustering.dataset import TabDataset

class OULADData(TabDataset):
    dataset_name = "OULAD"
    dataset_dir = "fair_clustering/raw_data/OULAD"
    file_name = "studentInfo.csv"

    # Define the default mappings for encoding sensitive attributes, and labels
    default_mappings = {
        'gender': {'M': 0, 'F': 1},
        'disability': {'N': 0, 'Y': 1},
        'final_result': {'Fail': 0, 'Pass': 1}
    }
    
    label_name = 'final_result'
    
    def __init__(self, center=True, exclude_s=True):
        file_path = os.path.join(self.dataset_dir, self.file_name)
        df = pd.read_csv(file_path)

        # Remove rows with missing attributes
        df.dropna(inplace=True)
        
        # Remove rows with 'final_result' as 'Withdrawn'
        df = df[df['final_result'] != 'Withdrawn']

        # Replace 'Distinction' in 'final_result' with 'Pass'
        df['final_result'].replace('Distinction', 'Pass', inplace=True)

        sensitive_attributes = 'gender'
        
        # Encoding for binary attribute 'disability'
        df['disability'] = df['disability'].map(self.default_mappings['disability'])

        # Call the super class __init__ with updated dataframe
        super(OULADData, self).__init__(
            df=df,
            sensitive=sensitive_attributes,
            exclude_s=exclude_s,
            center=center,
            numerical_s=False,
        )
    
    @property
    def avail_s(self):
        # Return a list of available sensitive attributes
        return ['gender']

    @property
    def categorical_feat(self):
        # Return a list of categorical features
        return ['code_module', 'code_presentation', 'region', 'highest_education', 'imd_band', 'age_band']

    @property
    def drop_feat(self):
        # Return a list of features to drop
        return ['id_student']

    @property
    def label_name(self):
        # Return the name of the label feature
        return 'final_result'
    
    @label_name.setter
    def label_name(self, value):
        self._label_name = value

if __name__ == "__main__":
    dataset = OULADData()
    X, y, s = dataset.data
    stat = dataset.stat
    print(stat)