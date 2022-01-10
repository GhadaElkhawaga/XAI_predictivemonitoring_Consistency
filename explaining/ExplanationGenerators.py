import csv
import os
import xgboost as xgb
import time
import graphviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.inspection import permutation_importance
from alibi.explainers import ALE, plot_ale
from explaining.ALEcomputations import feats_impurity


def Permutation_importance_analysis(artefacts_dir, cls, method_name, ffeatures,
                                    encoded_training, train_y_experiment, dataset_name,
                                    cls_method, bkt_size, prfx_len, feat_num):
    Permutation_dir = os.path.join(artefacts_dir, 'Permutation_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num))
    if not os.path.exists(Permutation_dir):
        os.makedirs(os.path.join(Permutation_dir))
    permutation_file_name = 'permutation_importance_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num)
    start_calc_train = time.time()
    training_result = permutation_importance(cls, encoded_training, train_y_experiment,
                                                 n_repeats=10, random_state=42, n_jobs=-1)
    perm_time_train = time.time() - start_calc_train
    cols = ['Feature', 'importance(mean)', 'importance(std)', 'importance']
    df_res_train = pd.DataFrame(zip(ffeatures, training_result.importances_mean,
                                    training_result.importances_std,
                                    training_result.importances), columns=cols)
    df_train_sorted = df_res_train.sort_values('importance(mean)', ascending=False)
    df_train_sorted.to_csv(os.path.join(Permutation_dir, '%s_training.csv'
                                        %permutation_file_name), sep=';', index=False)
    with open(os.path.join(Permutation_dir, '%s_training.csv' %permutation_file_name), 'a') as fout:
        fout.write('%s;%s\n' % ('Calculation time (Train)', perm_time_train))
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.boxplot(df_train_sorted.iloc[:20, 3].T, vert=False, labels=df_train_sorted.iloc[:20, 0])
    ax.set_title("Permutation Importance (train set)")
    plt.savefig(os.path.join(Permutation_dir, '%s_training.png' %permutation_file_name), dpi=300,
                bbox_inches='tight');
    plt.figure(figsize=(12, 8))
    plt.barh(np.arange(0, 20), df_train_sorted.iloc[:20, 1], align='center', alpha=0.5)
    plt.yticks(np.arange(0, 20), df_train_sorted.iloc[:20, 0])
    plt.xlabel('Importance')
    plt.title("Permutation Importance (train set)");
    plt.savefig(os.path.join(Permutation_dir, '%s_training2.png' %permutation_file_name), dpi=300,
                bbox_inches='tight');


def shap_global(artefacts_dir, cls, X, dataset_name, cls_method, method_name, ffeatures, bkt_size, prfx_len, feat_num,
                X_other=None, flag=None):
    shap_values_dir = os.path.join(artefacts_dir, 'shap_%s_%s_%s_%s_%s_%s' % (
    dataset_name, cls_method, method_name, bkt_size, prfx_len, feat_num))
    if not os.path.exists(shap_values_dir):
        os.makedirs(os.path.join(shap_values_dir))

    typ = flag
    shap_time = time.time()
    if shap.__version__ >= str(0.37):
        explainer = shap.Explainer(cls, X, feature_names=ffeatures)
    else:
        if cls_method == 'xgboost':
            explainer = shap.TreeExplainer(cls)
        else:
            explainer = shap.LinearExplainer(cls, X)

    if cls_method == 'xgboost':
            shap_values = explainer.shap_values(X, check_additivity=False)
    else:
            shap_values = explainer.shap_values(X)
    shap_time_end = time.time() - shap_time

    frmt_str = '%s_%s_%s_%s_%s_%s_%s' % (cls_method, dataset_name, method_name, typ, bkt_size, prfx_len, feat_num)

    out1 = os.path.join(shap_values_dir,
                        'shap_explainer_%s.pickle' % (frmt_str))
    with open(out1, 'wb') as output:
        pickle.dump(explainer, output)

    shap_data = os.path.join(shap_values_dir,
                             'shap_values_%s.pickle' % (frmt_str))
    with open(shap_data, 'wb') as fout:
        pickle.dump(shap_values, fout)

    shap_csv = os.path.join(shap_values_dir,
                            'shap_values_%s.csv' % (frmt_str))
    pd.DataFrame(shap_values).to_csv(shap_csv, sep=';', index=False)
    with open(shap_csv, 'a') as fout:
        fout.write('%s;%s\n' % ('Calculcation time', shap_time_end))


    if shap.__version__ >= str(0.37):
        shap.plots.beeswarm(shap_values, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, max_display=10, show=False)
    plt.savefig(os.path.join(shap_values_dir,
                             'Shap values_normal%s.png' % (frmt_str)),
                dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()

    if shap.__version__ >= str(0.37):
        shap.plots.bar(shap_values, show=False)
    else:
        shap.summary_plot(shap_values, X, feature_names=ffeatures, plot_type='bar', show=False, max_display=10)

    plt.savefig(
        os.path.join(shap_values_dir, 'Shap values_bar_%s.png' % (frmt_str)),
        dpi=300, bbox_inches='tight');
    plt.clf()
    plt.close()

    del explainer
    del shap_values

""""# Algorithm:
- examine the correlations dataset and compute ALE for features correlated with the target by more than 0.2
-  examine the correlations dataset and compute ALE for features correlated together by more than 0.35
- output a file with a list of features highly correlated with the target and a dictionary of features correlated with each other"""

def ALE_Computing(ALE_obj, bkt_size, prfx_len, feat_num):
    ALE_dir, ALE_df, ALE_training_arr, counts_df = ALE_obj.data_processing()
    ALE_features, target_names = ALE_obj.get_ALE_names(ALE_df)
    ALE_cls = ALE_obj.ALE_classifier(ALE_training_arr)
    ALE_df['target'] = ALE_obj.y
    ALE_df['target'].replace({1: ALE_obj.dm.pos_label, 0: ALE_obj.dm.neg_label}, inplace=True)
    frmt_str = '%s_%s_%s_%s' % (ALE_obj.dataset_name, ALE_obj.bkt_enc, bkt_size, prfx_len)
    pred_exp_start = time.time()
    ale_pred = ALE(ALE_cls.predict_proba, feature_names=ALE_features, target_names=target_names)

    explainer_pred = ale_pred.explain(ALE_training_arr)
    pred_exp_time = time.time() - pred_exp_start

    explainer_pred_data = os.path.join(ALE_dir, 'ALE_pred_explainer_%s_%s.pickle' % (ALE_obj.cls_method, frmt_str))
    with open(explainer_pred_data, 'wb') as output:
        pickle.dump(explainer_pred, output)

    # saving values at which ale values are computed for each feature in csv files:
    cols = ['Feature', 'ALE_vals', 'feature_values_for_ALECalc', 'ale0' ,'entropy']
    ent = feats_impurity(explainer_pred.feature_names, explainer_pred.ale_values)

    df_res_exp_pred = pd.DataFrame(
        zip(explainer_pred.feature_names, explainer_pred.ale_values,
            explainer_pred.feature_values,explainer_pred.ale0, pd.Series(ent)), columns=cols)
    df_res_exp_pred_file = os.path.join(ALE_dir, 'ALE_pred_explainer_%s_%s.csv'
                                        %(ALE_obj.cls_method, frmt_str))
    df_res_exp_pred_sorted = df_res_exp_pred.sort_values('entropy')
    df_res_exp_pred.to_csv(df_res_exp_pred_file, sep=';', index=False)
    with open(df_res_exp_pred_file, 'a') as fout:
        fout.write('%s;%s\n' % ('Calculcation time', pred_exp_time))

    plt.figure(figsize=(12, 8))
    plt.barh(np.arange(0, len(df_res_exp_pred_sorted['entropy'])), df_res_exp_pred_sorted.iloc[:, 4], align='center',
             alpha=0.5)
    plt.yticks(np.arange(0, len(df_res_exp_pred_sorted['entropy'])), df_res_exp_pred_sorted.iloc[:, 0])
    plt.xlabel('impurity')
    plt.title("quality of split");
    plt.savefig(os.path.join(ALE_dir, '%s_impuritybasedonALE.png' % (frmt_str)), dpi=300,
                bbox_inches='tight');
    plt.close()


