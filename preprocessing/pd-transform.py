''' Transform the folded data so that it can be used by pd'''
import pandas as pd
from os import listdir, makedirs
from os.path import exists, join

def main():
  new_cols = ['Start Time','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Label']
  old_cols = ['Start Time','0','0625','125','1875','25','3125','375','4375','5','5625','625','6875','75','8125','875','9375','Label']
  print(len(old_cols), len(new_cols))
  # new_cols = ['Input_A',  'Input_B', 'Input_C', 'Input_D', 'Input_E', 'Input_F', 'Input_H', 'Input_I', 'Input_J', 'Input_K', 'Input_L', 'Input_M', 'Input_N', 'Input_O', 'Input_P', 'Output']
  # loop through the directories
  fold_path = '../data/sampled/'
  pd_ready_path = '../data/pd-ready/'
  directories = listdir(fold_path)
  for directory in directories:
    files = listdir(join(fold_path,directory))
    save_folder = join(pd_ready_path,directory.replace(' ', '_'))
    if not exists(save_folder):
      makedirs(save_folder)
    for file in files:
      inpath = join(fold_path,directory,file)
      data = pd.read_csv(inpath)
      data.columns = new_cols
      # seperate the label from the data, save seperately
      if 'test' in file:
        labels = data.pop('Label')
        label_outpath = join(pd_ready_path, directory.replace(' ', '_'), rename(file, 'labels'))
        pd.DataFrame.to_csv(labels, label_outpath, index=False, sep=' ')
      
      feature_outpath = join(pd_ready_path, directory.replace(' ', '_'), rename(file, 'features'))
      pd.DataFrame.to_csv(data, feature_outpath, index=False, sep=' ')
      

def rename(str, suf):
  tokens = str.split('.')
  return "{}_{}.{}".format(tokens[0], suf, 'txt')

if __name__ == "__main__":
  main()