import pandas as pd
import ast
import string
import os
import time
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from preprocessing.datasets_preprocessing import discretize, scale_update
from preprocessing.Indispensable_feats_gen import compute_features_importance
from utils.retrieval import retrieve_datasets_info, retrieve_artefact, retrieve_vector
from utils.retrieval import get_imp_features
from modeling.params_optimisation import optimise_params
from modeling.Models_training import train_model_predict
from evaluation.Consistency_measures import calculate_abic
from evaluation.Consistency_ratios import compute_reducts_core, ComputeRatio


n_iter = 3
out_dir = 'XAI_global_FeatsConsis'
if not os.path.exists(os.path.join(out_dir)):
  os.makedirs(os.path.join(out_dir))
reds_core_dir = os.path.join(out_dir, 'reducts_core')
if not os.path.exists(reds_core_dir):
  os.makedirs(reds_core_dir)
measurements_dir = os.path.join(out_dir, 'measurements')
if not os.path.exists(measurements_dir):
  os.makedirs(measurements_dir)

for artefacts_dir in ['single_agg_xgboost', 'single_agg_logit',
                      'prefix_index_logit', 'prefix_index_xgboost']:
    if not os.path.exists(artefacts_dir):
        os.makedirs(os.path.join(artefacts_dir))
    params_dir = os.path.join(artefacts_dir, 'cv_results_revision')
    if not os.path.exists(params_dir):
        os.makedirs(os.path.join(params_dir))
    results_dir_final = os.path.join(artefacts_dir, 'final_experiments_results')
    if not os.path.exists(results_dir_final):
        os.makedirs(os.path.join(results_dir_final))

discretized_folder = 'discretized_datasets'
saved_artefacts = os.path.join('model_and_hdf5')
if not os.path.exists(saved_artefacts):
        os.makedirs(os.path.join(saved_artefacts))

for method_name in ['single_agg', 'prefix_index']:
    if method_name == 'single_agg':
         datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"hospital_billing_1","hospital_billing_2", "BPIC2017_O_Accepted", "BPIC2017_O_Cancelled", "BPIC2017_O_Refused"]
         gap = 1
    else:
         datasets = ["sepsis1", "sepsis2", "sepsis3",'traffic_fines',"BPIC2017_O_Refused", "BPIC2017_O_Accepted"]
         gap = 5
    training_info, _ = retrieve_datasets_info(saved_artefacts,'all_datasets_info.csv', datasets, method_name)
    discretize(training_info, method_name, discretized_folder)

    for dataset_name in datasets:
        for cls_method in ['xgboost', 'logit']:
            artefacts_dir = '%s_%s' % (method_name, cls_method)
            params_dir = os.path.join(artefacts_dir, 'cv_results_revision')
            results_dir_final = os.path.join(artefacts_dir, 'final_experiments_results')
            optimise_params(artefacts_dir, dataset_name, cls_method, method_name, params_dir)
            train_model_explain(saved_artefacts,artefacts_dir, params_dir, results_dir_final,
                                gap, n_iter, 'agg', dataset_name, method_name,cls_method)

        entry = training_info.loc[(training_info['dataset_name']== dataset_name)&
                                      (training_info['method']== method_name),:]
        for _, data in entry.iterrows():
                bkt_size = data['bkt_size']
                prfx_len = data['prfx_len']
                feat_num = data['feat_num']
                file_name = '%s_%s_%s_%s_%s' % (dataset_name, method_name,
                                                bkt_size, prfx_len, feat_num)
                fstr = '%s_%s_%s_%s' % (method_name, bkt_size, prfx_len, feat_num)
                encoded_data_file = retrieve_artefact('encoded_datasets_%s' % method_name, '.csv',
                                                          'encoded_training', file_name)
                ffeatures = list(encoded_data_file.columns[:-1])
                scaler = MinMaxScaler()
                X_train = pd.DataFrame(scaler.fit_transform(encoded_data_file.loc
                                                            [:, encoded_data_file.columns != 'encoded_label']))
                X_train.columns = ffeatures
                y_train = encoded_data_file['encoded_label'].copy()
                model_xgboost = retrieve_artefact(saved_artefacts, '.pickle', 'model_%s_%s' %('xgboost', file_name))
                model_logit = retrieve_artefact(saved_artefacts, '.pickle', 'model_%s_%s' %('logit', file_name))
                df_discretized = retrieve_artefact(discretized_folder, '.csv', 'discretized_dataset',
                                                   'encoded_training', file_name)
                feats_imp_start = time.time()
                compute_features_importance(out_dir, dataset_name, X_train, y_train, ffeatures,
                                            'encoded_label', model_xgboost, model_logit,
                                            df_discretized, file_name)
                feats_imp_time = time.time() - feats_imp_start
                indisp_feats_start = time.time()
                compute_reducts_core(out_dir, file_name, reds_core_dir)
                red_core_time = time.time() - indisp_feats_start
                coreset = retrieve_vector(reds_core_dir, 'core_%s_python.json' %file_name)
                if not coreset:
                    continue
                else:
                    core_Feats_indices = coreset
                    core_feats = [ffeatures[i] for i in coreset]
                all_reds = retrieve_vector(reds_core_dir, 'reds_%s_python.json' %file_name)
                lengths_list = [len(l) for l in all_reds.values() if len(l) != len(ffeatures)]
                shortest_red_idx = lengths_list.index(min(lengths_list))  # the shortest reduct is selected
                selected_red = list(all_reds.values())[shortest_red_idx]
                reduct_feats_indices = selected_red
                reduct_feats = [ffeatures[item] for item in selected_red]
                selected_by_xai_reduct , reduct_xai_time, selected_by_xai_core,\
                core_xai_time = (defaultdict(dict) for i in range(4))
                for cls_method in ['xgboost', 'logit']:
                    for xai_type in ['shap', 'perm', 'ALE']:
                        key = '%s_%s_%s' % (dataset_name, cls_method, xai_type)
                        out_folder = {'ALE': 'ALE_artefacts', 'shap': 'shap_artefacts', 'perm': 'perm_artefacts'}
                        core_xai_time_start = time.time()

                        selected_by_xai_core[key] = get_imp_features(out_folder[xai_type], file_name, dataset_name,
                                                                     ffeatures, cls_method,
                                                                     len(core_Feats_indices), fstr, xai_type)
                        core_xai_time[key] = time.time() - core_xai_time_start
                        reduct_xai_time_start = time.time()
                        selected_by_xai_reduct[key] = get_imp_features(out_folder[xai_type], file_name, dataset_name,
                                                                       ffeatures, cls_method,
                                                                       len(reduct_feats_indices), fstr, xai_type)
                        reduct_xai_time[key] = time.time() - reduct_xai_time_start

                    for x in [selected_by_xai_core, selected_by_xai_reduct]:
                        scale_update(x)
                    # calculate the AIC/BIC values for the core/reduct of the complete features set
                    feats_len = len(ffeatures)

                    reduct_ratio_dict, core_ratio_dict, AIC_reduct_dict, AIC_core_dict, \
                    BIC_reduct_dict, BIC_core_dict, interreduct_scores_sum, reduct_intersection, \
                    intercore_scores_sum, core_intersection, red_ratio_time, core_ratio_time\
                        = (defaultdict(dict) for i in range(12))

                    for key in selected_by_xai_reduct.keys():
                        # calculations using intersection features, and their scores:
                        red_time_start = time.time()
                        reduct_ratio_dict[key], interreduct_scores_sum[key], reduct_intersection[key] = \
                            ComputeRatio(selected_by_xai_reduct[key], reduct_feats, 'regular')
                        comp_intersect_reduct = len(reduct_feats_indices) - reduct_intersection[key]
                        AIC_reduct_dict[key] = calculate_abic(bkt_size, reduct_ratio_dict[key],
                                                              comp_intersect_reduct, 'AIC')
                        red_ratio_time[key] = time.time() - red_time_start
                        BIC_reduct_dict[key] = calculate_abic(bkt_size, reduct_ratio_dict[key],
                                                              comp_intersect_reduct, 'BIC')


                        core_time_start = time.time()
                        core_ratio_dict[key], intercore_scores_sum[key], core_intersection[key] = \
                            ComputeRatio(selected_by_xai_core[key], core_feats, 'regular')
                        comp_intersect_core = len(core_feats) - core_intersection[key]
                        AIC_core_dict[key] = calculate_abic(bkt_size, core_ratio_dict[key],
                                                            comp_intersect_core, 'AIC')
                        core_ratio_time[key] = time.time() - core_time_start
                        BIC_core_dict[key] = calculate_abic(bkt_size, core_ratio_dict[key],
                                                            comp_intersect_core, 'BIC')


                    measurements = ['reduct_ratio', 'core_ratio', 'AIC_reduct',
                                    'AIC_core', 'BIC_reduct', 'BIC_core',
                                    'intersect_reduct_scores_sum', '#feats_reduct_intersection',
                                    'intersect_core_scores_sum', '#feats_core_intersection',
                                    'red_ratio_time', 'core_ratio_time']
                    results_df = pd.DataFrame.from_dict([reduct_ratio_dict, core_ratio_dict,
                                                         AIC_reduct_dict, AIC_core_dict,
                                                         BIC_reduct_dict, BIC_core_dict,
                                                         interreduct_scores_sum, reduct_intersection,
                                                         intercore_scores_sum, core_intersection,
                                                         red_ratio_time, core_ratio_time])
                    results_df = pd.concat([pd.DataFrame(measurements), results_df], axis=1)
                    results_df.columns.values[0] = 'Measurement'
                    results_df.to_csv(os.path.join(measurements_dir, 'measurements_%s.csv' % (file_name)),
                                      sep=';', index=False)
                    with open(os.path.join(measurements_dir, 'measurements_%s.csv' % (file_name)), 'a') as fout:
                        fout.write('\n')
                        fout.write('%s;%s\n' % ('Reduct_feats_%s' % (file_name), reduct_feats))
                        fout.write('%s;%s\n' % ('Core_feats_%s' % (file_name), core_feats))
                        fout.write('\n')
                    selected_feats_reduct = pd.DataFrame.from_dict([selected_by_xai_reduct])
                    xai_reduct_time = pd.DataFrame.from_dict([reduct_xai_time])
                    selected_feats_core = pd.DataFrame.from_dict([selected_by_xai_core])
                    xai_core_time = pd.DataFrame.from_dict([core_xai_time])
                    selected_feats_df = pd.concat([selected_feats_reduct, selected_feats_core,
                                                   xai_reduct_time, xai_core_time], axis=0)
                    selected_feats_df.reset_index(inplace=True, drop=True)
                    idx = pd.DataFrame(['selected_by_xai_reduct', 'selected_by_xai_core',
                                        'reduct_xai_time', 'core_xai_time'])
                    selected_feats_df = pd.concat([idx, selected_feats_df], axis=1)
                    selected_feats_df.to_csv(os.path.join(measurements_dir, 'measurements_%s.csv' % (file_name)),
                                             sep=';', mode='a', index=False)
                with open(os.path.join(measurements_dir, 'measurements_%s.csv' % (file_name)), 'a') as fout:
                    fout.write('%s;%s\n' % ('Feats_importance_computation_time_%s' % (file_name), feats_imp_time))
                    fout.write('%s;%s\n' % ('Reduct_core_computation_time_%s' % (file_name), red_core_time))
                    fout.write('\n')
