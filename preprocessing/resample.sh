train_path=../data/folds
label=Label
majority=Low
out_path=../data/sampled


NN=nearest_neighbor
NB=naive_bayes
RU=undersampling
GN=gan
SM=oversampling

cd $train_path
feature_list=(*)
cd -
for feature in "${feature_list[@]}"
do
  for i in {0..3}
  do
    # mkdir -p $out_path/$NN
    # python3 $strat_path/$NN.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$NN/balanced_${i}.csv

    # mkdir -p $out_path/$NB
    # python3 $strat_path/$NB.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$NB/balanced_${i}.csv

    mkdir -p $out_path/${feature}
    python3 $RU.py $train_path/${feature}/train_fold_${i}.csv -l $label -M Low -o $out_path/${feature}/balanced_${i}.csv

    # mkdir -p $out_path/$GN
    # python3 $strat_path/$GN.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$GN/balanced_${i}.csv

    # mkdir -p $out_path/$SM
    # python3 $strat_path/$SM.py $train_path${i}.csv -l $label -M 3 -m 4 -o $out_path/$SM/balanced_${i}.csv
  done
done