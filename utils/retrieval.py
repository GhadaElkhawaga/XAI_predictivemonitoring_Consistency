import pickle
import pandas as pd
from scipy.io import arff
import json
import os
import numpy as np
from numpy import percentile
import Definitions


def retrieve_dataFrame(grouped_info, method_name, datasets):
    subject_datasets_dict = {"dataset_name": [], "method": [], "bkt_size": [], "prfx_len": [], "feat_num": []};

    for idx, group in grouped_info:
        if idx[1] == method_name and idx[0] in datasets:
            if (method_name == 'single_agg'):
                gap = 1
            else:
                gap = 5
            for i in range(1, group.shape[0] + 1, gap):
                for row_idx, row in group.iterrows():
                    if i == row['prefix_length'] and row['bucket_size'] > 500:
                        subject_datasets_dict['dataset_name'].append(idx[0])
                        subject_datasets_dict['method'].append(idx[1])
                        subject_datasets_dict['bkt_size'].append(row['bucket_size'])
                        subject_datasets_dict['prfx_len'].append(row['prefix_length'])
                        subject_datasets_dict['feat_num'].append(row['feature_num'])

    return pd.DataFrame.from_dict(subject_datasets_dict)


def retrieve_datasets_info(dir, datasets_info, datasets, method_name):
    """
    a function to retrieve information about datasets from the all_datasets_info file containing relevant information
    """
    info_df = pd.read_csv(os.path.join(dir, datasets_info), sep=';')
    # to drop rows containing info about training datasets:
    training_info_df = info_df[info_df.dataset_type.str.contains("training")]
    info_df = info_df[~info_df.dataset_type.str.contains("training")]
    info_df.drop(['dataset_type'], inplace=True, axis=1)
    testing_grouped_info = info_df.groupby(['dataset', 'method'])
    training_grouped_info = training_info_df.groupby(['dataset', 'method'])

    training_info = retrieve_dataFrame(training_grouped_info, method_name, datasets)
    testing_info = retrieve_dataFrame(testing_grouped_info, method_name, datasets)
    return training_info, testing_info


def retrieve_dataset_cols(dataset_name, required):
    case_id_col = Definitions.case_id_col[dataset_name]
    activity_col = Definitions.activity_col[dataset_name]
    timestamp_col = Definitions.timestamp_col[dataset_name]
    label_col = Definitions.label_col[dataset_name]
    pos_label = Definitions.pos_label[dataset_name]
    neg_label = Definitions.neg_label[dataset_name]
    dynamic_cat_cols = Definitions.dynamic_cat_cols[dataset_name]
    static_cat_cols = Definitions.static_cat_cols[dataset_name]
    dynamic_num_cols = Definitions.dynamic_num_cols[dataset_name]
    static_num_cols = Definitions.static_num_cols[dataset_name]

    if required == 'basic':
        return [case_id_col, activity_col, timestamp_col]
    elif required == 'target':
        return [label_col, pos_label, neg_label]
    elif required == 'cols':
        return [dynamic_cat_cols, static_cat_cols, dynamic_num_cols, static_num_cols]

def retrieve_vector(folder, fname):
    with open(os.path.join(folder,fname)) as f:
        return json.load(f)


def retrieve_artefact(folder, file_end, *argv):
    retrieved_file = retrieve_file(folder, file_end, argv)
    if '.pickle' in file_end:
        with open(retrieved_file, 'rb') as fin:
            retrieved_artefact = pickle.load(fin)
    else:
        retrieved_artefact = pd.read_csv(retrieved_file, sep=';', encoding='ISO-8859-1')
    return retrieved_artefact


# a function to retrieve files of artefacts
def retrieve_file(folder, file_end, argv):
    sep = '_'
    file_name = sep.join([str(a) for a in argv])
    file_name += file_end
    return os.path.join(folder, file_name)


# a function to get important features according to each XAI method:
def get_imp_features(folder, file_name, dataset_name, ffeatures, cls_method, num_retrieved_feats, fstr, xai_type=None):

    frmt_str = {
        'ALE': 'ALE_pred_explainer_%s_%s.csv' % (cls_method, file_name),
        'shap': 'shap_values_%s_%s.csv' % (cls_method, file_name),
        'perm': 'permutation_importance_%s_%s_%s_final.csv' %(
    dataset_name, cls_method, fstr)}
    # number of features to be retrieved from the imp features set
    if xai_type != None:
        feats_df = retrieve_artefact(folder, '.csv', frmt_str[xai_type])
        local_feats = num_retrieved_feats
        feats_df.drop(feats_df.tail(1).index, inplace=True)
        if xai_type == 'ALE':
            feats_df_sorted = pd.DataFrame(feats_df, columns=feats_df.columns)
            imp_feats_scores = feats_df_sorted.iloc[:local_feats, -1]
            # adjust local_feats if the number of features passed to ALE is different than the whole feature set
            local_feats = min(num_retrieved_feats, len(feats_df_sorted.iloc[:, 0]))
        elif xai_type == 'shap':
            avg_shap_values = pd.DataFrame(np.mean(np.abs(feats_df.values), 0), columns=['shap_vals'])
            feats_df = pd.concat([pd.Series(ffeatures), avg_shap_values], axis=1)
            feats_df_sorted = feats_df.sort_values(by='shap_vals', ascending=False)
        elif xai_type == 'perm':
            feats_df_sorted = pd.DataFrame(data=feats_df.values, columns=feats_df.columns)
        else:
            raise ValueError('XAI method not found')

    imp_feats_names = feats_df_sorted.iloc[:local_feats, 0]
    if xai_type in ['shap', 'perm']:
        imp_feats_scores = feats_df_sorted.iloc[:local_feats, 1]
    imp_feats = dict(zip(imp_feats_names, imp_feats_scores))
    return imp_feats




