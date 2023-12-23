import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import exists
pd.options.mode.chained_assignment = None  # default='warn'
## going to loop through to find zeros
# if a row is all zeros, we will drop it
# if a row has some zeros, we will use linear interpolation
def main():
# for each feature
  features = ['Angle Neck Rotation', 'Angle Neck Flex/Ext',	'Angle Neck Lateral flexion',
              'Angle Shoulder (Left) Vertical rotation','Angle Shoulder (Left) Horizontal rotation',	'Angle Shoulder (Left) Rotation',
              'Angle Shoulder (Left) (Projection) Flex/Ext', 'Angle Shoulder (Left) (Projection) Abd/Add',	'Angle Shoulder (Left) (Projection) Rotation',
              'Angle Shoulder (Right) Vertical rotation',	'Angle Shoulder (Right) Horizontal rotation',	'Angle Shoulder (Right) Rotation',
              'Angle Shoulder (Right) (Projection) Flex/Ext',	'Angle Shoulder (Right) (Projection) Abd/Add', 'Angle Shoulder (Right) (Projection) Rotation',	'Angle Back Forward flexion',
              'Angle Back Lateral flexion', 'Angle Back Rotation'
              ]
  for feature in features:
    feature_name = feature.replace('/', '-')

    data_path = f'../data/windowed/{feature_name}/'
    windowed_files = listdir(data_path)

    out_path = f'../data/cleaned/{feature_name}/'
    if not exists(out_path):
      makedirs(out_path)
        
    for filename in windowed_files:
      data = pd.read_csv(data_path+filename)
      # remove any full 0 or nan rows
      data.replace(0, np.nan, inplace=True)
      filtered_data = data.dropna(subset=data.columns[1:-1], how='all') # loc[(data.iloc[:, 2:-1] != np.nan).any(axis=1)]
      time_col = filtered_data.pop('Start Time')
      time_col.fillna(0, inplace=True)
      label_col = filtered_data.pop('Label')

      # convert values to numeric dtype
      columns = filtered_data.columns[filtered_data.dtypes.eq('object')]
      filtered_data[columns] = filtered_data[columns].apply(pd.to_numeric, errors='coerce')

      # convert values to numeric, then replace 0s with NaN to set up interpolation
      # unclean_data = filtered_data.iloc[:, 1:-1]
      # unclean_data.replace(0, np.nan, inplace=True)

      # interpolate remaining zeros
      clean_data = filtered_data.interpolate(
          method='linear', limit_direction='both', axis=1)
      
      # stitch data back together
      clean_data.insert(0, 'Start Time', time_col)
      clean_data['Label'] = label_col
      # save
      pd.DataFrame.to_csv(clean_data, out_path+filename.replace('Windowed', 'Cleaned'), index=False)
if __name__ == '__main__':
  main()