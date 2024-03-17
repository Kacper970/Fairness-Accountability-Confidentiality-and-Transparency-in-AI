import os
import numpy as np
from scipy.io import arff
import pandas as pd
from fair_clustering.dataset import TabDataset

class DutchCensusData(TabDataset):
    dataset_name = "Dutch_Census_2001"
    dataset_dir = "fair_clustering/raw_data/dutch_census"
    file_name = "dutch_census_2001.arff"

    # Define the default mappings for encoding sensitive attributes, and labels
    default_mappings = {
        'occupation': {'5_4_9': 0, '2_1': 1},
        'sex': {0: 0, 1: 1}
    }
    
    label_name = 'occupation'
    
    def __init__(self, center=True, exclude_s=True):
        file_path = os.path.join(self.dataset_dir, self.file_name)
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)

        # Adjust to 0 or 1 encoding for 'sex'
        df['sex'] = df['sex'].astype(int) - 1  

        # Replace 9 with NaN in 'prev_residence_place'
        df['prev_residence_place'] = df['prev_residence_place'].replace(9, np.nan)

        # Handle 'occupation' as a label
        df['occupation'] = df['occupation'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        self.y = df['occupation'].map({'5_4_9': 0, '2_1': 1}).to_numpy()

        # Define sensitive attribute
        sensitive_attributes = 'sex'

        # Call the super class __init__ with updated dataframe
        super(DutchCensusData, self).__init__(
            df=df,
            sensitive=sensitive_attributes,
            exclude_s=exclude_s,
            center=center,
            numerical_s=False,
        )
    
    @property
    def avail_s(self):
        # Return a list of available sensitive attributes
        return ['sex']

    @property
    def categorical_feat(self):
        # Return a list of categorical features
        return ['age', 'household_position', 'household_size', 'citizenship', 'country_birth', 'edu_level', 'economic_status', 'cur_eco_activity', 'Marital_status']

    @property
    def drop_feat(self):
        # Return a list of features to drop
        return []

    @property
    def label_name(self):
        # Return the name of the label feature
        return 'occupation'
    
    @label_name.setter
    def label_name(self, value):
        self._label_name = value

if __name__ == "__main__":
    dataset = DutchCensusData()
    X, y, s = dataset.data
    stat = dataset.stat
    print(stat)