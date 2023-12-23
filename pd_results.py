import pandas as pd
import matplotlib.pyplot as plt
from statistics import variance
from os.path import exists, join
from os import makedirs
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def main():
  # load in files
  # load in testing folds and take the 
  features = ['Angle_Neck_Rotation', 'Angle_Shoulder_(Left)_Horizontal_rotation',
              'Angle_Neck_Flex-Ext', 'Angle_Shoulder_(Left)_(Projection)_Abd-Add',
              # 'Angle_Shoulder_(Left)_Vertical_rotation',  'Angle_Shoulder_(Left)_Rotation', 'Angle_Neck_Lateral_flexion',
              # 'Angle_Shoulder_(Left)_(Projection)_Flex-Ext', ,'Angle_Shoulder_(Left)_(Projection)_Rotation',
              # 'Angle_Shoulder_(Right)_Vertical_rotation',	'Angle_Shoulder_(Right)_Horizontal_rotation',	'Angle_Shoulder_(Right)_Rotation',
              # 'Angle_Shoulder_(Right)_(Projection)_Flex-Ext',	'Angle_Shoulder_(Right)_(Projection)_Abd-Add', 'Angle_Shoulder_(Right)_(Projection)_Rotation',
              'Angle_Back_Forward_flexion', 'Angle_Back_Lateral_flexion', 'Angle_Back_Rotation'
              ]
  feature_map = {}
  result_path = 'results/pd/predictions'
  truth_path = 'data/folds/'
  for feature in features:
    print(feature)
    result_map = {}
    fold_accuracy = []
    fold_f1 = []
    predictions = []
    truths = []
    for fold in range(4):
      print(fold)
      pred_file = join(result_path, f'{feature}_{fold}.txt')
      y_pred = pd.read_csv(pred_file, sep=' ')['Unnamed: 0'].to_list()

      truth_file = join(truth_path, feature, f'test_fold_{fold}.csv')
      y_test = pd.read_csv(truth_file)['Label'].iloc[:len(y_pred)].to_list()

      # print(predictions, truths)
      fold_accuracy.append(accuracy_score(y_test, y_pred))
      fold_f1.append(f1_score(y_test, y_pred, average='macro'))

      predictions += y_pred
      truths +=  y_test
    
      conf_mat = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
      cm = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low', 'Medium', 'High'])
      if not exists(f'results/pd/cms/{feature}'):
        makedirs(f'results/pd/cms/{feature}')
      cm.plot(cmap=plt.cm.Blues).figure_.savefig(f'results/pd/cms/{feature}/fold_{fold}.png')
      plt.close()

    result_map['Accuracy'] = fold_accuracy
    result_map['AVariance'] = variance(fold_accuracy)
    result_map['F1'] = fold_f1
    result_map['FVariance'] = variance(fold_f1)

    feature_map[feature] = result_map

    conf_mat = confusion_matrix(truths, predictions, labels=['Low', 'Medium', 'High'])
    cm = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low', 'Medium', 'High'])
    cm.plot(cmap=plt.cm.Purples).figure_.savefig(f'results/pd/cms/{feature}/confusion_matrix.png')

  with open('./results/pd/metrics.json', 'w') as file:
    json.dump(feature_map, file)

if __name__ == '__main__':
  main()