import numpy as np
import pandas as pd
import os
from utils.retrieval import retrieve_datasets_info, retrieve_artefact, retrieve_vector
from utils.retrieval import get_imp_features
from preprocessing.datasets_preprocessing import scale_update
from evaluation.Consistency_measures import calculate_abic
from evaluation.Consistency_ratios import ComputeRatio
from collections import defaultdict


out_dir = 'XIA_global_FeatsConsis'
reds_core_dir = os.path.join(out_dir, 'reducts_core')
measurements_dir = os.path.join(out_dir, 'measurements')
experiments_dir = os.path.join(out_dir, 'experimental_measurements')
if not os.path.exists(experiments_dir):
  os.makedirs(experiments_dir)
saved_artefacts = os.path.join('model_and_hdf5')


# the same as the previous cell, but just for experimental reasons we use using intersection features and their count
for method_name in ['single_agg', 'prefix_index']:
    if method_name == 'single_agg':
        datasets = ["sepsis1", "sepsis2", "sepsis3", 'traffic_fines', "hospital_billing_1", "hospital_billing_2",
                    "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"]
        gap = 1
    else:
        datasets = ["sepsis1", "sepsis2", "sepsis3", 'traffic_fines', "BPIC2017_O_Refused", "BPIC2017_O_Accepted"]
        gap = 5
    training_info, _ = retrieve_datasets_info(saved_artefacts, 'all_datasets_info.csv', datasets, method_name)
    for dataset_name in datasets:
        entry = training_info.loc[(training_info['dataset_name']== dataset_name)&
                                      (training_info['method']== method_name),:]
        for _, data in entry.iterrows():
                bkt_size = data['bkt_size']
                prfx_len = data['prfx_len']
                feat_num = data['feat_num']
                file_name = '%s_%s_%s_%s_%s' % (dataset_name, method_name,
                                                bkt_size, prfx_len, feat_num)
                str = '%s_%s_%s' % (method_name, bkt_size, prfx_len)
                encoded_data_file = retrieve_artefact('encoded_datasets_%s' % method_name, '.csv',
                                                          'encoded_training', file_name)
                ffeatures = encoded_data_file.columns[:-1]


                coreset = retrieve_vector(reds_core_dir, 'core_%s_python.json' %(file_name))
                core_Feats_indices = coreset
                core_feats = [ffeatures[i] for i in coreset]

                all_reds = retrieve_vector(reds_core_dir, 'reds_%s_python.json' % (file_name))
                lengths_list = [len(l) for l in all_reds.values() if len(l) != len(ffeatures)]
                shortest_red_idx = lengths_list.index(min(lengths_list))  # the shortest reduct is selected
                selected_red = list(all_reds.values())[shortest_red_idx]
                reduct_feats_indices = selected_red
                reduct_feats = [ffeatures[item] for item in selected_red]

                selected_by_xai_reduct = {}
                selected_by_xai_core = {}
                for cls_method in ['xgboost', 'logit']:
                    for xai_type in ['shap', 'perm', 'ALE']:
                        key = '%s_%s_%s' % (dataset_name, cls_method, xai_type)
                        out_folder = {'ALE': 'ALE_artefacts', 'shap': 'shap_artefacts', 'perm': 'perm_artefacts'}
                        selected_by_xai_core[key] = get_imp_features(out_folder[xai_type], str, dataset_name,
                                                                     feat_num, ffeatures, cls_method,
                                                                     len(core_Feats_indices), xai_type)
                        selected_by_xai_reduct[key] = get_imp_features(out_folder[xai_type], str, dataset_name,
                                                                       feat_num, ffeatures, cls_method,
                                                                       len(reduct_feats_indices), xai_type)
                for x in [selected_by_xai_core, selected_by_xai_reduct]:
                    scale_update(x)
                '''calculate the AIC/BIC values for the core/reduct of the complete features set'''
                feats_len = len(ffeatures)

                reduct_ratio_exp_dict, core_ratio_exp_dict, AIC_reduct_exp_dict, AIC_core_exp_dict, \
                BIC_reduct_exp_dict, BIC_core_exp_dict, reduct_intersection, core_intersection = \
                    (defaultdict(dict) for i in range(8))

                for key in selected_by_xai_reduct.keys():
                    # just for experimenting and plotting reasons (using intersection features and their count):
                    reduct_ratio_exp_dict[key], reduct_intersection[key] = \
                        ComputeRatio(selected_by_xai_reduct[key], reduct_feats,'experimental')
                    comp_intersect_reduct = len(reduct_feats_indices) - reduct_intersection[key]
                    AIC_reduct_exp_dict[key] = \
                        calculate_abic(bkt_size, reduct_ratio_exp_dict[key], comp_intersect_reduct, 'AIC')
                    BIC_reduct_exp_dict[key] = \
                        calculate_abic(bkt_size, reduct_ratio_exp_dict[key], comp_intersect_reduct, 'BIC')
                    core_ratio_exp_dict[key], core_intersection[key] = \
                        ComputeRatio(selected_by_xai_core[key], core_feats, 'experimental')
                    comp_intersect_core = len(core_feats) - core_intersection[key]
                    AIC_core_exp_dict[key] = \
                        calculate_abic(bkt_size, core_ratio_exp_dict[key], comp_intersect_core, 'AIC')
                    BIC_core_exp_dict[key] = \
                        calculate_abic(bkt_size, core_ratio_exp_dict[key], comp_intersect_core, 'BIC')

                measurements = ['reduct_ratio', 'core_ratio', 'AIC_reduct',
                                'AIC_core', 'BIC_reduct', 'BIC_core',
                                '#feats_reduct_intersection', '#feats_core_intersection']

                results_df = pd.DataFrame.from_dict([reduct_ratio_exp_dict, core_ratio_exp_dict,
                                                     AIC_reduct_exp_dict, AIC_core_exp_dict,
                                                     BIC_reduct_exp_dict, BIC_core_exp_dict,
                                                     reduct_intersection, core_intersection])
                results_df = pd.concat([pd.DataFrame(measurements), results_df], axis=1)
                results_df.columns.values[0] = 'Measurement'

                # consistency_df = pd.DataFrame()
                results_df.to_csv(os.path.join(experiments_dir, 'measurements_experimental_%s.csv' %file_name),
                                  sep=';', index=False)
                with open(os.path.join(experiments_dir, 'measurements_experimental_%s.csv' % file_name), 'a') as fout:
                    fout.write('\n')
                    fout.write('%s;%s\n' % ('Reduct_feats_%s' %file_name, reduct_feats))
                    fout.write('%s;%s\n' % ('Core_feats_%s' %file_name, core_feats))
                    fout.write('\n')

                selected_feats_reduct = pd.DataFrame.from_dict([selected_by_xai_reduct])
                selected_feats_core = pd.DataFrame.from_dict([selected_by_xai_core])
                selected_feats_df = pd.concat([selected_feats_reduct, selected_feats_core], axis=0)
                selected_feats_df.reset_index(inplace=True, drop=True)
                idx = pd.DataFrame(['selected_by_xai_reduct', 'selected_by_xai_core'])

                selected_feats_df = pd.concat([idx, selected_feats_df], axis=1)
                selected_feats_df.to_csv(os.path.join(experiments_dir, 'measurements_experimental_%s.csv' %file_name),
                                         sep=';', mode='a', index=False)

    for dataset_name in datasets:
        entry = training_info.loc[(training_info['dataset_name'] == dataset_name) &
                                  (training_info['method'] == method_name), :]
        for _, data in entry.iterrows():
            bkt_size = data['bkt_size']
            prfx_len = data['prfx_len']
            feat_num = data['feat_num']
            file_name = '%s_%s_%s_%s_%s' % (dataset_name, method_name,
                                                    bkt_size, prfx_len, feat_num)
        measurements = retrieve_artefact(measurements_dir, 'measurements_%s' %file_name,
                                                 '.csv', ';')
        measurements_experiments = retrieve_artefact(experiments_dir,
                                                             'measurements_experimental_%s' %file_name,
                                                             '.csv', ';')

        header = ['Measurement', '%s_xgboost_shap' %dataset_name, '%s_xgboost_perm' %dataset_name,
                          '%s_xgboost_ALE' %dataset_name, '%s_logit_shap' %dataset_name,
                          '%s_logit_perm' %dataset_name, '%s_logit_ALE' %dataset_name]
        exp_l = []
        exp_l.append(measurements[measurements['Measurement'] == '#feats_reduct_intersection'])
        exp_l.append(measurements_experiments[measurements_experiments['Measurement'] == 'AIC_reduct'])
        exp_l.append(measurements[measurements['Measurement'] == 'intersect_reduct_scores_sum'])
        exp_l.append(measurements[measurements['Measurement'] == 'AIC_reduct'])
        exp_l.append(measurements[measurements['Measurement'] == '#feats_core_intersection'])
        exp_l.append(measurements_experiments[measurements_experiments['Measurement'] == 'AIC_core'])
        exp_l.append(measurements[measurements['Measurement'] == 'intersect_core_scores_sum'])
        exp_l.append(measurements[measurements['Measurement'] == 'AIC_core'])
        exp_df = pd.DataFrame(np.concatenate(exp_l), columns=header)
        with open(os.path.join(experiments_dir, 'exp_res.csv'), 'a') as f:
            pd.concat([exp_df]).to_csv(f, sep=';', index=False)
            f.write('\n')

