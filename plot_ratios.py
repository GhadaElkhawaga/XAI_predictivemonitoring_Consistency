import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from utils.retrieval import retrieve_artefact, retrieve_datasets_info
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def plot(x,y,title_str,cls,marker):
  plt.plot(x, y, label = "%s_%s" %(title_str,cls),marker=marker,markersize=5)
  return


out_dir = os.path.join('XAI_global_FeatsConsis')
in_dir = os.path.join(out_dir,'measurements')
plots = os.path.join(out_dir,'plots')
if not os.path.exists(plots):
    os.makedirs(plots)
saved_artefacts = os.path.join('model_and_hdf5')
if not os.path.exists(saved_artefacts):
        os.makedirs(os.path.join(saved_artefacts))
l = [] 
training_info = pd.DataFrame()
for method_name in ['prefix_index', 'single_agg']:
  if method_name == 'single_agg':
    datasets = ["sepsis1", "sepsis2", "sepsis3"]
    training_info_method, _ = retrieve_datasets_info(saved_artefacts,'all_datasets_info.csv', datasets, method_name)
  else:
    datasets = [ "sepsis2", 'traffic_fines', "BPIC2017_O_Accepted"]      
    training_info_method, _ = retrieve_datasets_info(saved_artefacts,'all_datasets_info.csv', datasets, method_name)
  training_info = pd.concat([training_info, training_info_method])
training_info.reset_index(drop=True, inplace=True)

for ratio in ['reduct', 'core']:
  for cls_name in ['xgboost', 'logit']:
    for xai_method in ['shap','perm','ALE']:
      l.append('%s_%s_%s'%(ratio, cls_name, xai_method))

rows = []
for _, data in training_info.iterrows():
  dataset_name = data['dataset_name']
  method_name = data['method']
  bkt_size = data['bkt_size']
  prfx_len = data['prfx_len']
  feat_num = data['feat_num']
  file_name = '%s_%s_%s_%s_%s' % (dataset_name, method_name, bkt_size, prfx_len, feat_num)
  measurements_df = retrieve_artefact(in_dir, '.csv', 'measurements_%s' %file_name).iloc[0:2,:]
  new_list = list(map(float,[*measurements_df.iloc[0,1:], *measurements_df.iloc[1,1:]]))
  rows.append(new_list)

ratios_df = pd.DataFrame(rows, columns=l)
ratios_df.reset_index(inplace=True, drop=True)
training_ratios_df = pd.concat([training_info, ratios_df], axis = 1) 
training_ratios_df.to_csv(os.path.join(plots, 'training_ratios.csv'), sep=';')


grouped_training = training_ratios_df.groupby(['dataset_name', 'method'], as_index=False)  
for _, grp in grouped_training:
    if 'single_agg' in str(grp['method']):
        rratios_names = [x.replace('reduct_','R_') for x in grp.columns[5:11]]
        cratios_names = [x.replace('core_','C_') for x in grp.columns[11:]]
        rratios = grp.iloc[0,5:11].values
        cratios = grp.iloc[0,11:].values
        plt.ylabel('Ratios')
        plt.xticks(rotation=90, fontsize=5)
        plt.yticks(fontsize=5)
        plt.bar(rratios_names, rratios, width = 0.95, label = 'Reduct Ratios')
        for i, rr in enumerate(rratios):
            plt.text(x = rratios_names[i],  y = rr , s = str(round(rr,5)),va = 'bottom', ha = 'center', fontdict=dict(fontsize=5))
        plt.bar(cratios_names, cratios, width = 0.95, label = 'Core Ratios')
        for i, cr in enumerate(cratios):
            plt.text(x = cratios_names[i], y = cr, s = str(round(cr,5)), va = 'bottom', ha = 'center', fontdict=dict(fontsize=5))
        plt.legend()
        file_name = '_'.join(str(elm) for elm in grp.iloc[0,0:5].values)
        plt.savefig(os.path.join(plots,'Ratios_%s.png'%(file_name)), dpi=300, bbox_inches='tight');
        plt.clf();
    else:
        x = grp.iloc[:,3]  #prfx lengths
        colors = ['red','blue','purple','darkgreen','orange','navy','mediumvioletred','maroon','steelblue','crimson','black', 'peru']
        for i in range(5,len(grp.columns)):
            y = grp.iloc[:,i]
            plt.plot(x,y, color = colors[i-5], label = grp.columns[i], marker='o')
            for xs, ys in zip(x,y):
                l = '%s' %(str(round(ys,5)))
        plt.legend(fontsize=5)
        plt.grid(linestyle = '--', linewidth = 0.5)
        file_name = '_'.join(str(elm) for elm in grp.iloc[0,0:2].values)
        plt.savefig(os.path.join(plots,'Ratios_%s.png'%(file_name)), dpi=300, bbox_inches='tight');
        plt.clf();


  


