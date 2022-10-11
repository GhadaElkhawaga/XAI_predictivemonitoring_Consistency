import pandas as pd
import os
from utils.retrieval import retrieve_artefact
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer

def discretize(training_info, method_name, discretized_folder):
    for i, r in training_info.iterrows():
        total_bins = []
        file_name = r.str.cat(sep="_")
        df = retrieve_artefact('encoded_datasets_%s' % (method_name), file_name, '.csv', vals_sep=';')
        X_df = df[df.columns[:-1]].copy()
        y_series = df[df.columns[-1]].copy()
        for i in X_df.columns:
            if len(X_df[i].value_counts()) >= 10:
                bins = 4
            else:
                bins = len(X_df[i].value_counts())
            total_bins.append(bins)
        try:
            discretizer = KBinsDiscretizer(n_bins=min(total_bins)+1, encode='ordinal', strategy='uniform')
            X_df = pd.DataFrame(discretizer.fit_transform(X_df), columns=header)
            df = pd.concat([X_df, y_series], axis=1, join='inner')
            df.to_csv(os.path.join(discretized_folder, 'discretized_dataset_%s.csv' % (file_name)),
                      sep=',', index=False)
        except:
            continue
    return



def scale_update(old_dict):
    scale_df = pd.DataFrame()
    for x in old_dict.keys():
      vals = old_dict[x]
      scale_df[x] = vals.values()
    for col in scale_df.columns.tolist():
        scale_df[col] = MinMaxScaler().fit_transform(scale_df[col].values.reshape(-1, 1))
    for i in range(len(old_dict.keys())):
      keys = list(list(old_dict.values())[i].keys())
      scaled_vals = list(scale_df[scale_df.columns[i]].values)
      unscaled_dict = list(old_dict.values())[i]
      dictionary = dict(zip(keys,scaled_vals))
      unscaled_dict = unscaled_dict.update(dictionary)
    return
