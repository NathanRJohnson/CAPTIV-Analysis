#!/bin/bash

# calls the child and train and test for each fold of each feature
# creates unique file names to capture the results of each run

script_path=~/4910/
output_path=njohns18/testResults/
data_path=~/4910/data/pd-ready
configs_path=~/4910/results/pd/configs
results_path=~/4910/results/pd/predictions

chmod +x pd_child_train.sh pd_child_test.sh
mkdir results/pd
mkdir $configs_path
mkdir $results_path

cd $data_path

feature_list=(*)
# Loop through the array elements
for feature in "${feature_list[@]}"
do
    # echo "Feature: $feature"
    cd ${feature} 
    file_list=(*)
    for i in {0..3}
    do
        echo $i
        # get the training file for this fold
        train_file=$(ls | grep "^train.*$i.*$")
        # echo $train_file

        # get the testing file for this fold
        test_file=$(ls | grep "^test.*${i}_features.txt$")
        # echo $test_file
        
        config_file=${feature}_${i}.cfg
        predicitions_file=${feature}_${i}.txt
        # train pattern discovery
        cd $script_path
        # echo ${configs_path}/${config_file}
        ./pd_child_train.sh $configs_path/$config_file $data_path/$feature/$train_file
        # get + store the results
        ./pd_child_test.sh $configs_path/$config_file $results_path/$predicitions_file $data_path/$feature/$test_file

        cd $data_path/$feature
    done
    exit
    cd $data_path
done