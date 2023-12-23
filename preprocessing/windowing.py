# AE1_03 05/12/22 05/19/22  LOW
# AE1_18 05/17/2022	05/24/2022 LOW
# AE1_19 05/13/2022	05/20/2022 LOW
# AE1_20 05/30/2022	06/30/2022 LOW
# AE4_23	DBG (Week 2)	06/20/2022	06/28/2022 LOW
# AE4_24	DBG(Week 3)	07/07/2022	07/12/2022 LOW
# AE1_25	DBG (Week 3)	7/22/2022	7/28/2022 LOW

import pandas as pd
import os.path
import os

def main():
  # setup paths ------------------------------
  data_path = '../data/'
  participant_path = data_path + 'participant/'

  header_size = 135
  # setup meta data --------------------------
  window_size = 16 # window has 16 entries
  interval_length = 2 # pick every 2nd row
  skip_length = 16 # skip half a second

  # Select paticipant ------------------------
  participants = [
                  'AE1_03', 
                  'AE1_04', 'AE1_06', 'AE1_07', 'AE1_09', 'AE1_10', 'AE1_19', 'AE1_20',
                  'AE2_08', 'AE3_12', 'AE3_13', 
                  'AE3_14', 'AE3_16', 'AE3_17', 'AE4_23', 'AE4_24'
                 ]
  parts = ['Part 1', 'Part 2', 'Part 3']
  labels = {'AE1_03':'Low', 'AE1_04': 'Medium', 'AE1_06': 'Low', 'AE1_07': 'Low', 'AE1_09': 'Low', 'AE1_10':'Medium',
            'AE1_19': 'Low', 'AE1_20': 'Low', 'AE2_08': 'Medium', 'AE3_12':'Low',
              'AE3_13':'High', 'AE3_14':'Low', 'AE3_16':'Low', 'AE3_17':'Low', 'AE4_23': 'Low', 'AE4_24': 'Low'}
  features = ['Angle Neck Rotation', 'Angle Neck Flex/Ext',	'Angle Neck Lateral flexion', 
              'Angle Shoulder (Left) Vertical rotation', 'Angle Shoulder (Left) Horizontal rotation',	'Angle Shoulder (Left) Rotation',	
              'Angle Shoulder (Left) (Projection) Flex/Ext', 'Angle Shoulder (Left) (Projection) Abd/Add',	'Angle Shoulder (Left) (Projection) Rotation',
              'Angle Shoulder (Right) Vertical rotation',	'Angle Shoulder (Right) Horizontal rotation',	'Angle Shoulder (Right) Rotation',
              'Angle Shoulder (Right) (Projection) Flex/Ext',	'Angle Shoulder (Right) (Projection) Abd/Add', 'Angle Shoulder (Right) (Projection) Rotation',	'Angle Back Forward flexion',
              'Angle Back Lateral flexion', 'Angle Back Rotation']

  for participant in participants:
    print(participant)
    # Select feature ---------------------------
    for feature in features:
      windowed_df = pd.DataFrame(columns=create_column_headers(window_size))
      i=0
      new_index=1
      time = 0.0
      # print(feature)

      for part in parts:
        filename = f'{participant} DBG {part} Angles.csv'
        # print(filename)
        if not os.path.isfile(participant_path+filename):
          continue

        participant_data = pd.read_csv(participant_path + filename, sep='\t', encoding='utf-16', header=getHeaderSize(participant))
        participant_data = participant_data.drop(columns=participant_data.columns[25:], axis=1)
        num_rows = participant_data.shape[0]
        # Get participant's label

        try: # AE3_14 doesn't have any back data....
          window_column = participant_data[feature]
        except:
          continue
        # Process the data --------------------------
        new_row = []
        while i < num_rows:
          # print(i)
          new_row.clear()
          new_row.append(time)
          
          if ((num_rows-1) - i < window_size * interval_length):
            break

          for col_j in range(window_size):
            new_row.append(window_column[i + interval_length*col_j])
          new_row.append(labels[participant])
          windowed_df.loc[new_index] = new_row
        
          new_index += 1
          i += skip_length
          time += 0.5

      feature_name = feature.replace('/', '-')
      outpath = data_path + f'windowed/{feature_name}/'
      if not os.path.exists(outpath):
        os.makedirs(outpath)
      outfile = outpath + f'{participant} Windowed - {feature_name} - {window_size} {interval_length} {skip_length}.csv'
      # print(f'Saving to {outfile}')
      pd.DataFrame.to_csv(windowed_df, outfile, index=False)
 
def create_column_headers(num_columns):
  nums = []
  for i in range(0, 16):
    nums.append(i/16)

  columns = ['Start Time']
  columns += ['+'+str(i) for i in nums]
  columns.append('Label')
  return columns

def getHeaderSize(participant_id):
  header_size = 135
  if 'AE3' in participant_id:
    if '13' in participant_id or '17' in participant_id:
      header_size = 145
    elif '14' in participant_id:
      header_size = 90
  return header_size

if __name__ == "__main__":
  main()
