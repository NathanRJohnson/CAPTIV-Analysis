import pandas as pd
from os import listdir, makedirs
from os.path import exists
## We want between participants
## Since it's time series data, we want to keep the data in order i.e not use the future to predict the past

def main():
  # for a specific feature
  # select a train size or test_size
  features = ['Angle Neck Rotation', 'Angle Neck Flex/Ext',	'Angle Neck Lateral flexion',
              'Angle Shoulder (Left) Vertical rotation', 'Angle Shoulder (Left) Horizontal rotation',	'Angle Shoulder (Left) Rotation',
  'Angle Shoulder (Left) (Projection) Flex/Ext', 'Angle Shoulder (Left) (Projection) Abd/Add',	'Angle Shoulder (Left) (Projection) Rotation',
  'Angle Shoulder (Right) Vertical rotation',	'Angle Shoulder (Right) Horizontal rotation',	'Angle Shoulder (Right) Rotation',
  'Angle Shoulder (Right) (Projection) Flex/Ext',	'Angle Shoulder (Right) (Projection) Abd/Add', 'Angle Shoulder (Right) (Projection) Rotation',	'Angle Back Forward flexion',
  'Angle Back Lateral flexion', 'Angle Back Rotation']
  for feature in features:
    feature_name = feature.replace('/', '-')

    # 16 participants 4 per fold. 12 test 4 train
    # 17 columns, 16 for the window times, 1 for the label
    train_folds = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    test_folds = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    # open a file, add it to 3 of 4  
    data_path = f'../data/cleaned/{feature_name}/'
    cleaned_files = listdir(data_path)

    for i, filename in enumerate(cleaned_files):
      test_fold_id = i % 4
      participant_data = pd.read_csv(data_path+filename)
      if 'AE3_14' in filename and feature in ['Angle Back Forward flexion', 'Angle Back Lateral flexion', 'Angle Back Rotation']:
        continue

      for j in range(0,4):
        if test_fold_id == j: # test set for this fold
          test_folds[j] = pd.concat([test_folds[j], participant_data], ignore_index=True)
        else: # train set
          train_folds[j] = pd.concat([train_folds[j], participant_data], ignore_index=True)

    file_feature_name = feature_name.replace(' ', '_')
    fold_path = f'../data/folds/{file_feature_name}/'
    if not exists(fold_path):
      makedirs(fold_path)
    for i in range(4):
      train_fold_name = f'train_fold_{i}.csv'
      test_fold_name = f'test_fold_{i}.csv'
      pd.DataFrame.to_csv(train_folds[i], fold_path+train_fold_name, index=False)
      pd.DataFrame.to_csv(test_folds[i], fold_path+test_fold_name, index=False)
    

if __name__ == '__main__':
  main()