import json
import pandas as pd
import numpy as np
import math 
def main():
  with open('results/btree/metrics.json', 'r') as file:
    bt_results = json.load(file)

  bt_table = make_table(bt_results)
  print('--- Base Tree ---\n', bt_table[:5], '\n\n')
  
  pd.DataFrame.to_csv(bt_table, 'bt_table.csv')
  
  
  with open('results/ptree/metrics.json', 'r') as file:
    pt_results = json.load(file)

  pt_table = make_table(pt_results)
  print('--- Pruned Tree ---\n', pt_table[1:6], '\n\n')
  print(pt_table[1:].mean())
  pd.DataFrame.to_csv(pt_table, 'pt_table.csv')


  with open('results/svm/metrics.json', 'r') as file:
    svc_results = json.load(file)
  
  svc_table = make_table(svc_results)
  print('--- Support Vector Machine ---\n\n', svc_table[:5], '\n\n')
  # print(svc_table.mean())
  pd.DataFrame.to_csv(svc_table, 'svc_table.csv')



  with open('results/pd/metrics.json', 'r') as file:
    pd_results = json.load(file)
  
  pd_table = make_table(pd_results)
  print('--- Pattern Discovery ---\n\n', pd_table[:5])
  pd.DataFrame.to_csv(pd_table, 'pd_table.csv')

  # print(pd_table.mean())

def make_table(data):
  table = pd.DataFrame()
  features = ['Angle_Neck_Rotation', 'Angle_Neck_Flex-Ext',	'Angle_Neck_Lateral_flexion',
              'Angle_Shoulder_(Left)_Vertical_rotation', 'Angle_Shoulder_(Left)_Horizontal_rotation', 'Angle_Shoulder_(Left)_Rotation',
              'Angle_Shoulder_(Left)_(Projection)_Flex-Ext', 'Angle_Shoulder_(Left)_(Projection)_Abd-Add',	'Angle_Shoulder_(Left)_(Projection)_Rotation',
              'Angle_Shoulder_(Right)_Vertical_rotation',	'Angle_Shoulder_(Right)_Horizontal_rotation',	'Angle_Shoulder_(Right)_Rotation',
              'Angle_Shoulder_(Right)_(Projection)_Flex-Ext',	'Angle_Shoulder_(Right)_(Projection)_Abd-Add', 'Angle_Shoulder_(Right)_(Projection)_Rotation',
              'Angle_Back_Forward_flexion', 'Angle_Back_Lateral_flexion', 'Angle_Back_Rotation'
              ]
  metrics = ['Accuracy', 'F1']
  for feature in features:
    if not feature in data.keys():
      continue
    vals, vars = split_dict(data[feature])
    row = []
    for metric in metrics:
      metric_mean = np.mean(vals[metric])
      row.append(metric_mean)      
    for key in vars.keys():
      row.append(vars[key])
    table[feature] = row
  column_headings = {i:features[i] for i in range(len(features))}
  table.rename(columns=column_headings)
  table = table.transpose()
  row_indices = metrics + ['Acc Var', 'F1 Var'] 
  table.columns = row_indices
  # trunc = lambda x: math.trunc(1000 * x) / 1000;
  table = table.round(3)
  return table.sort_values(by='F1', axis=0, ascending=False)

def split_dict(original_dict):
  keys_for_dict1 = ['Accuracy', 'F1']
  keys_for_dict2 = ['AVariance', 'FVariance']

  dict1 = {key: original_dict[key] for key in keys_for_dict1 if key in original_dict}
  dict2 = {key: original_dict[key] for key in keys_for_dict2 if key in original_dict}

  return dict1, dict2

if __name__ == '__main__':
  main()

