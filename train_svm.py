from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from statistics import variance
import pandas as pd
import numpy as np
import json
from os.path import join, exists
from os import makedirs

def convert_lab(lab):
  lab_map = {'Low': 0, 'Medium': 1, 'High': 2}
  return lab_map.get(lab)

def main():
  feature_map = {}
  group_accuracy = {0: [], 1:[], 2:[], 3:[]}
  group_f1 = {0: [], 1:[], 2:[], 3:[]}
  features = ['Angle_Neck_Rotation', 'Angle_Neck_Flex-Ext',	'Angle_Neck_Lateral_flexion',
              'Angle_Shoulder_(Left)_Vertical_rotation', 'Angle_Shoulder_(Left)_Horizontal_rotation', 'Angle_Shoulder_(Left)_Rotation',
              'Angle_Shoulder_(Left)_(Projection)_Flex-Ext', 'Angle_Shoulder_(Left)_(Projection)_Abd-Add','Angle_Shoulder_(Left)_(Projection)_Rotation',
              'Angle_Shoulder_(Right)_Vertical_rotation',	'Angle_Shoulder_(Right)_Horizontal_rotation',	'Angle_Shoulder_(Right)_Rotation',
              'Angle_Shoulder_(Right)_(Projection)_Flex-Ext',	'Angle_Shoulder_(Right)_(Projection)_Abd-Add', 'Angle_Shoulder_(Right)_(Projection)_Rotation',
              'Angle_Back_Forward_flexion', 'Angle_Back_Lateral_flexion', 'Angle_Back_Rotation'
              ]
  all_true = []
  all_pred = []
  for feature in features:
    print(feature)
    result_map = {}
    train_path = f'data/sampled/{feature}/'
    test_path = f'data/folds/{feature}/'

    fold_accuracy = []
    fold_f1 = []

    predictions = []
    truths = []
    for i in range(4):
      training_data = pd.read_csv(join(train_path, f'balanced_{i}.csv'))
      x_train = training_data.iloc[:,1:-1]
      y_train = training_data["Label"]
      if (len(x_train) == 0):
        continue

      testing_data = pd.read_csv(join(test_path, f'test_fold_{i}.csv'))
      x_test = testing_data.iloc[:, 1:-1]
      y_test = testing_data["Label"]

      clf = SVC(kernel='poly', degree=3, max_iter=10000)
      clf.fit(x_train, y_train)

      y_pred = clf.predict(x_test)
      acc = accuracy_score(y_test, y_pred)
      fold_accuracy.append(acc)
      group_accuracy[i].append(acc)
      fscore = f1_score(y_test, y_pred, average='macro')
      fold_f1.append(fscore)
      group_f1[i].append(fscore)

      predictions += y_pred.tolist()
      truths += y_test.tolist()

      conf_mat = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
      cm = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low', 'Medium', 'High'])
      if not exists(f'results/svm/cms/{feature}'):
        makedirs(f'results/svm/cms/{feature}')
      cm.plot(cmap=plt.cm.Blues).figure_.savefig(f'results/svm/cms/{feature}/fold_{i}.png')
      plt.close()

    result_map['Accuracy'] = fold_accuracy
    result_map['AVariance'] = variance(fold_accuracy)
    result_map['F1'] = fold_f1
    result_map['FVariance'] = variance(fold_f1)

    feature_map[feature] = result_map
    all_true += truths
    all_pred += predictions
    conf_mat = confusion_matrix(truths, predictions, labels=['Low', 'Medium', 'High'])
    cm = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low', 'Medium', 'High'])
    cm.plot(cmap=plt.cm.Purples).figure_.savefig(f'results/svm/cms/{feature}/confusion_matrix.png')

    # print final values
  conf_mat = confusion_matrix(all_true, all_pred, labels=['Low', 'Medium', 'High'])
  cm = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low', 'Medium', 'High'])
  cm.plot(cmap=plt.cm.Purples).figure_.savefig(f'results/svm/cms/confusion_matrix.png')  
  
  group_acc_means = []
  group_f1_means = []
  for j in range(4):
    group_acc_means.append(np.mean(group_accuracy[j]))
    group_f1_means.append(np.mean(group_f1[j]))

  group_map = {}
  group_map['GAccuracy'] = group_acc_means
  group_map['GF1'] = group_f1_means
  with open('./results/svm/groups.json', 'w+') as file:
    json.dump(group_map, file)

  with open('./results/svm/metrics.json', 'w') as file:
    json.dump(feature_map, file)

if __name__ == '__main__':
  main()