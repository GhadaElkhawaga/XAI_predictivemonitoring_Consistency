
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import Definitions

#logs_dir = 'logs'


class DatasetManager:

    def __init__(self, d_name):
        self.d_name = d_name
        self.case_id_col = Definitions.case_id_col[self.d_name]
        self.activity_col = Definitions.activity_col[self.d_name]
        self.timestamp_col = Definitions.timestamp_col[self.d_name]
        self.label_col = Definitions.label_col[self.d_name]
        self.pos_label = Definitions.pos_label[self.d_name]
        self.neg_label = Definitions.neg_label[self.d_name]

        self.dynamic_cat_cols = Definitions.dynamic_cat_cols[self.d_name]
        self.static_cat_cols = Definitions.static_cat_cols[self.d_name]
        self.dynamic_num_cols = Definitions.dynamic_num_cols[self.d_name]
        self.static_num_cols = Definitions.static_num_cols[self.d_name]

        self.sorting_cols = [self.timestamp_col, self.activity_col]

    def read_dataset(self):
        dtypes = {col: 'object' for col in (self.dynamic_cat_cols + self.static_cat_cols + [
            self.case_id_col + self.label_col + self.timestamp_col])}
        for col in (self.dynamic_num_cols + self.static_num_cols):
            dtypes[col] = 'float'

        # read encoded data
        df = pd.read_csv(Definitions.filename[self.d_name], sep=';', dtype=dtypes, engine='c', encoding='ISO-8859-1',
                         error_bad_lines=False)
        return df
    
    
    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]
        
        
    # to determine the length of a case which has a positive outcome, used in determining the max prefix length for each log
    def get_pos_case_length_quantile(self, data, percentage=0.90):
        return int(
            np.ceil(data[data[self.label_col] == self.pos_label].groupby(self.case_id_col).size().quantile(percentage)))



    # to get the ratio of samples belonging to the positive class
    def get_class_ratio(self, data):
        frequencies = data[self.label_col].value_counts()
        return frequencies[self.pos_label] / frequencies.sum()

    def get_label_numeric(self, data):
        # get the label of the first row in a process instance, as they are grouped
        y = data.groupby(self.case_id_col).first()[self.label_col]
        return [1 if label == self.pos_label else 0 for label in y]


    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()['prefix_nr']
