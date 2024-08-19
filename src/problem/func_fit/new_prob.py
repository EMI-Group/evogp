import numpy as np

from .func_fit import FuncFit

import pandas as pd

class New_Prob(FuncFit):

    def __init__(self, target_name):
        super().__init__()
        self.target_name = target_name

    @property
    def inputs(self):
        file_path = '/home/kelvin/data/x-t.csv'
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        return data.iloc[:, :15].values.astype(np.float32)
        
    @property
    def targets(self):
        file_path = '/home/kelvin/data/x-t.csv'
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        return data[self.target_name].values.reshape(-1,1).astype(np.float32)

    @property
    def input_shape(self):
        return 848, 15

    @property
    def output_shape(self):
        return 848, 1