import pickle
import ast
import os
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from explaining.XAImethods import explain_predictions
from evaluation.Consistency_measures import calculate_abic
from helpers.Encoders import get_encoder
from helpers.Bucketers import get_bucketer
from helpers.DatasetManager import DatasetManager
from explaining.ExplanationGenerators import Permutation_importance_analysis, shap_global, ALE_Computing


def train_model_explain(saved_artefacts, artefacts_dir, params_dir, results_dir_final, gap, n_iter, bucket_encoding,
                                dataset_name, method_name,cls_method):
    resexpfile = os.path.join(artefacts_dir, 'results_LIME_%s_%s_%s.csv' % (cls_method, dataset_name, method_name))
    explanationfile = os.path.join(artefacts_dir,
                                   'explanationsfile_LIME_%s_%s_%s.csv' % (cls_method, dataset_name, method_name))
    # fileEmpty = os.stat(explanationfile).st_size == 0
    with open(explanationfile, 'w+') as expf:
        header = ['Dataset Name', 'case ID', 'Actual Value', 'Explanation', 'Probability result', 'Predicted class',
                  'Class Type', 'Generation Time', 'Coefficient Stability', 'Variable Stability']
        writer = csv.DictWriter(expf, delimiter=';', lineterminator='\n', fieldnames=header)
        # if fileEmpty:
        writer.writeheader()

    with open(resexpfile, 'w+') as resf:
        header2 = ['Case ID', 'Explanation', 'Label']
        writer2 = csv.DictWriter(resf, delimiter=';', lineterminator='\n', fieldnames=header2)
        # if fileEmpty:
        writer2.writeheader()

    params_file = os.path.join(params_dir, 'optimal_params_%s_%s_%s.pickle' % (cls_method, dataset_name, method_name))
    with open(params_file, 'rb') as fin:
        args = pickle.load(fin)


    current_args = {}

    dm = DatasetManager(dataset_name)
    df = dm.read_dataset()

    cls_encoder_args_final = {'case_id_col': dm.case_id_col,
                              'static_cat_cols': dm.static_cat_cols,
                              'dynamic_cat_cols': dm.dynamic_cat_cols,
                              'static_num_cols': dm.static_num_cols,
                              'dynamic_num_cols': dm.dynamic_num_cols,
                              'fillna': True}

    # determine min and max (truncated) prefix lengths
    min_prefix_length_final = 1
    if "traffic_fines" in dataset_name:
        max_prefix_length_final = 10
    elif "BPIC2017" in dataset_name:
        max_prefix_length_final = min(20, dm.get_pos_case_length_quantile(df, 0.90))
    else:
        max_prefix_length_final = min(40, dm.get_pos_case_length_quantile(df, 0.90))

    train, test = dm.split_data_strict(df, train_ratio=0.8, split='temporal')

    if gap > 1:
        outfile = os.path.join(results_dir_final, "performance_experiments_%s_%s_%s_gap%s.csv" % (
            cls_method, dataset_name, method_name, gap))
    else:
        outfile = os.path.join(results_dir_final,
                               'performance_experiments_%s_%s_%s.csv' % (cls_method, dataset_name, method_name))

    prefix_test_generation_start = time.time()
    if method_name == 'prefix_index':
        df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final, gap=5)
    else:
        df_test_prefixes = dm.generate_prefix_data(test, min_prefix_length_final, max_prefix_length_final)
    prefix_test_generation = time.time() - prefix_test_generation_start

    train_prefix_generation_times = []
    offline_times = []
    online_times = []
    for i in range(n_iter):
        print('starting Iteration number {0} in file {1}'.format(i, dataset_name))
        train_prefix_generation_start = time.time()
        if method_name == 'prefix_index':
            df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final, gap=5)
        else:
            df_train_prefixes = dm.generate_prefix_data(train, min_prefix_length_final, max_prefix_length_final)
        train_prefix_generation = time.time() - train_prefix_generation_start
        train_prefix_generation_times.append(train_prefix_generation)

        # Bucketing prefixes based on control flow
        bucketer_args = {'encoding_method': bucket_encoding,
                         'case_id_col': dm.case_id_col,
                         'cat_cols': [dm.activity_col],
                         'num_cols': [],
                         'random_state': random_state}

        bucketer = get_bucketer(bucket_method, **bucketer_args)

        start_offline_bucket = time.time()
        bucket_assignments_train = bucketer.fit_predict(df_train_prefixes)
        offline_bucket = time.time() - start_offline_bucket

        bucket_assignment_test = bucketer.predict(df_test_prefixes)

        preds_all = []
        test_y_all = []

        nr_events_all = []
        offline_time_fit = 0
        current_online_times = []

        for bucket in set(bucket_assignment_test):
            if "prefix" in method_name:
                current_args = args[bucket]
            else:
                current_args = args

            relevant_train_bucket = dm.get_indexes(df_train_prefixes)[bucket == bucket_assignments_train]
            relevant_test_bucket = dm.get_indexes(df_test_prefixes)[bucket == bucket_assignment_test]
            df_test_bucket = dm.get_data_by_indexes(df_test_prefixes, relevant_test_bucket)
            test_prfx_len = dm.get_prefix_lengths(df_test_bucket)[0]

            nr_events_all.extend(list(dm.get_prefix_lengths(df_test_bucket)))
            if len(relevant_train_bucket) == 0:
                preds = [dm.get_class_ratio(train)] * len(relevant_test_bucket)
                current_online_times.extend([0] * len(preds))
            else:
                df_train_bucket = dm.get_data_by_indexes(df_train_prefixes, relevant_train_bucket)
                train_y_experiment = dm.get_label_numeric(df_train_bucket)
                prfx_len = dm.get_prefix_lengths(df_train_bucket)[0]

                if len(set(train_y_experiment)) < 2:
                    preds = [train_y_experiment[0]] * len(relevant_train_bucket)
                    current_online_times.extend([0] * len(preds))
                    test_y_all.extend(dm.get_label_numeric(df_test_prefixes))
                else:
                    start_offline_time_fit = time.time()
                    featureCombinerExperiment = FeatureUnion(
                        [(method, get_encoder(method, **cls_encoder_args_final)) for method in methods])

                    if cls_method == 'xgboost':
                        cls_experiment = xgb.XGBClassifier(objective='binary:logistic',
                                                           n_estimators=500,
                                                           learning_rate=current_args['learning_rate'],
                                                           max_depth=current_args['max_depth'],
                                                           subsample=current_args['subsample'],
                                                           colsample_bytree=current_args['colsample_bytree'],
                                                           min_child_weight=current_args['min_child_weight'],
                                                           seed=random_state)

                        pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('cls', cls_experiment)])
                    else:
                        cls_experiment = LogisticRegression(C=2 ** current_args['C'], random_state=random_state)
                        pipeline_final = Pipeline([('encoder', featureCombinerExperiment), ('scaler', StandardScaler()),
                                                   ('cls', cls_experiment)])

                    pipeline_final.fit(df_train_bucket, train_y_experiment)
                    offline_time_train_single_bucket = time.time() - start_offline_time_fit

                    offline_time_fit += time.time() - start_offline_time_fit
                    with open(outfile, 'a') as out:
                        out.write('%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                            'dataset', 'method', 'cls', 'nr_events', 'n_iter', 'prefix_length',
                            'train_time_bucket','score'))
                        out.write('%s;%s;%s;%s;%s;%s;%s;%s\n' % (
                            dataset_name, method_name, cls_method, -1, i, prfx_len,
                            offline_time_train_single_bucket, -1))

                    if (i == 2):

                        ffeatures = pipeline_final.named_steps['encoder'].get_feature_names()
                        encoded_training = featureCombinerExperiment.fit_transform(df_train_bucket)
                        training_set_df = pd.DataFrame(encoded_training, columns=ffeatures)
                        bkt_size = training_set_df.shape[0]
                        feat_num = training_set_df.shape[1]
                        # save the features names for later use
                        ffeatures_file = os.path.join(saved_artefacts,
                                                      'ffeatures_{0}_{1}_{2}_{3}_{4}.pickle'.format(cls_method,
                                                                                                    dataset_name,
                                                                                                    method_name,
                                                                                                    bkt_size, prfx_len,
                                                                                                    feat_num))
                        with open(ffeatures_file, 'wb') as fout_features:
                            pickle.dump(ffeatures, fout_features)

                        model_saved = pipeline_final.named_steps['cls']
                        # save the model for later use
                        model_file = os.path.join(saved_artefacts, 'model_%s_%s_%s_%s_%s_%s.pickle' % (
                            cls_method, dataset_name, method_name, bkt_size, prfx_len, feat_num))
                        with open(model_file, 'wb') as fout:
                            pickle.dump(model_saved, fout)

                        if (get_percentage(1, train_y_experiment) > 0.1) or (
                                get_percentage(0, train_y_experiment) > 0.1):
                            explain_flag = True

                        if (explain_flag == True):
                            encoded_testing_bucket = featureCombinerExperiment.fit_transform(df_test_bucket)
                            testing_set_df = pd.DataFrame(encoded_testing_bucket, columns=ffeatures)
                            test_bkt_size = testing_set_df.shape[0]

                            shap_global(artefacts_dir, cls_experiment, encoded_training,
                                        dataset_name, cls_method, method_name, ffeatures, bkt_size,
                                        prfx_len, feat_num, X_other=encoded_testing_bucket,
                                        flag='training')

                    preds = []
                    test_y_bucket = []
                    test_buckets_grouped = df_test_bucket.groupby(dm.case_id_col)



                    for idx, grouppred in test_buckets_grouped:

                        test_y_all.extend(dm.get_label_numeric(grouppred))
                        if method_name == 'prefix_index':
                            test_y_bucket.extend(dm.get_label_numeric(grouppred))
                        start_prediction = time.time()
                        preds_pos_label_idx = np.where(cls_experiment.classes_ == 1)[0][0]
                        pred = pipeline_final.predict_proba(grouppred)[:, preds_pos_label_idx]
                        pipeline_final_prediction_time = time.time() - start_prediction
                        current_online_times.append(pipeline_final_prediction_time / len(grouppred))
                        preds.extend(pred)

                    if (i == 2) and (explain_flag == True):
                        y_real = test_y_bucket if method_name == 'prefix_index' else test_y_all
                        Permutation_importance_analysis(artefacts_dir, pipeline_final.named_steps['cls'],
                                                        method_name, ffeatures, encoded_training,
                                                        train_y_experiment, dataset_name,
                                                        cls_method, bkt_size, prfx_len, feat_num)

                        ALE_obj = ALEProcessing(artefacts_dir, dataset_name, cls_method, method_name, dm, ffeatures,
                                                encoded_training, train_y_experiment, cls_experiment)
                        ALE_obj_file = os.path.join(saved_artefacts, 'ALEObj_%s_%s_%s_%s_%s_%s.pickle' % (
                            cls_method, dataset_name, method_name, bkt_size, prfx_len, feat_num))
                        with open(ALE_obj_file, 'wb') as fout:
                            pickle.dump(ALE_obj, fout)
                        ALE_Computing(ALE_obj, bkt_size, prfx_len, feat_num)


            preds_all.extend(preds)

        offline_total_time = offline_bucket + offline_time_fit + train_prefix_generation
        offline_times.append(offline_total_time)
        online_times.append(current_online_times)

    with open(outfile, 'w') as out:
        out.write('%s;%s;%s;%s;%s;%s;%s\n' % ('dataset', 'method', 'cls', 'nr_events', 'n_iter', 'metric', 'score'))
        out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
            dataset_name, method_name, cls_method, -1, -1, 'test_prefix_generation_time', prefix_test_generation))

        for j in range(len(offline_times)):
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'train_prefix_generation_time', train_prefix_generation))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'offline_bucket_time', offline_bucket))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'total_offline_time', offline_times[j]))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'online_time_average', np.mean(online_times[j])))
            out.write('%s;%s;%s;%s;%s;%s;%s\n' % (
                dataset_name, method_name, cls_method, -1, j, 'online_time_standard', np.std(online_times[j])))

        df_results = pd.DataFrame({'actual': test_y_all, 'predicted': preds_all, 'nr_events': nr_events_all})
        for nr_events, group in df_results.groupby('nr_events'):
            if len(set(group.actual)) < 2:
                out.write(
                    "%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan))
            else:
                try:
                    out.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc",
                                                          roc_auc_score(group.actual, group.predicted)))
                except ValueError:
                    pass
        try:
            out.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, -1, "auc",
                                                  roc_auc_score(df_results.actual, df_results.predicted)))
        except ValueError:
            pass

        online_event_times_flat = [t for iter_online_event_times in online_times for t in iter_online_event_times]
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat)))
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat)))
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_times)))
        out.write("%s;%s;%s;%s;%s;%s;%s\n" % (
            dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_times)))