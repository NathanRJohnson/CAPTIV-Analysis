#!/bin/bash

# Child script which runs the Pattern Discovery models using the supplied
# training and testing data.

# read in cfg paht
config_path=$1

# read in the path to the training data
train_path=$2

# echo "t: $train_path, $config_path"

cd pd
./pdTrain +veOnly -S2 -o$config_path -lLabel $train_path
cd - > /dev/null

# pdTrain -odata/xor/xor100.cfg -lOutput data/xor/xor100_train.txt

