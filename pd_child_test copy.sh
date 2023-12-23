#!/bin/bash

# Child script which runs the Pattern Discovery models using the supplied
# training and testing data.

# read in cfg paht
config_path=$1

# read in the path to the training data
output_path=$2

test_path=$3

# echo "t: $test_path, $config_path"

cd pd
./pdClass -l -c$config_path -o$output_path $test_path
cd - > /dev/null

# pdTrain -odata/xor/xor100.cfg -lOutput data/xor/xor100_train.txt

