import pandas as pd
from statistics import variance
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def run_clf(clf, feature):

  accuracy_by_fold = []
  f1_by_fold = []

  # load data
  data_path = f'data/folds/{feature}/'

  for i in range(4):
    training_data = pd.read_csv(data_path+f'train_fold_{i}.csv')
    x_train = training_data.iloc[:,1:-1].to_numpy().tolist()
    y_train = training_data["Label"].to_numpy().tolist()
    testing_data = pd.read_csv(data_path+f'test_fold_{i}.csv')
    x_test = testing_data.iloc[:, 1:-1].to_numpy().tolist()
    y_test = testing_data["Label"].to_numpy().tolist()

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy_by_fold.append(accuracy_score(y_test, y_pred))
    f1_by_fold.append(f1_score(y_test, y_pred, average='macro'))

    predictions += y_pred.tolist()
    truths += y_test.tolist()

  mean_accuracy = sum(accuracy_by_fold) / len(accuracy_by_fold)
  var_accuracy = variance(accuracy_by_fold)
  mean_f1 = sum(f1_by_fold) / len(f1_by_fold)
  var_f1 = variance(f1_by_fold)
  conf_mat = confusion_matrix(truths, predictions, labels=['Low', 'Medium', 'High'])

  results = {'mean accuracy': mean_accuracy, 
             'var_accuracy': var_accuracy,
             'mean_f1': var_accuracy,
             'var_f1': var_f1,
             'matrix': conf_mat
            }
  return results

def main():
  pass

if __name__ == '__main__':
  main()