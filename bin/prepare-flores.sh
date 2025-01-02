#!/usr/bin/env bash
# Author: Suzanna Sia

RUN_MODE=(0 1)
DATAD=$(pwd)/data

# RUN_MODE=0: download the dataset
# RUN_MODE=1: split dev dataset into a train(800) and prompt(200) split.

og_file=https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz

for run in ${RUN_MODE[@]}; do
	if [ $run -eq 0 ]; then
		mkdir -p $DATAD/FLORES 
		wget $og_file -O $DATAD/FLORES/flores101_dataset.tar.gz
		cd $DATAD/FLORES
		tar zxvf flores101_dataset.tar.gz
		rm flores101_dataset.tar.gz
	fi

	if [ $run -eq 1 ]; then
		dev_dir=$DATAD/FLORES/flores101_dataset/dev
		cp -r $dev_dir $DATAD/FLORES/flores101_dataset/dev_raw
		mkdir -p $DATAD/FLORES/flores101_dataset/train_split

		fns=$(ls $dev_dir)
		for fn in ${fns[@]}; do
				echo $fn
				sed -n '1,800p' $dev_dir/$fn > $DATAD/FLORES/flores101_dataset/train_split/$fn
				sed -n '801,1000p' $dev_dir/$fn > $dev_dir/$fn.tmp
				mv $DATAD/FLORES/flores101_dataset/dev/$fn.tmp $DATAD/FLORES/flores101_dataset/dev/$fn
		done
	fi
done
