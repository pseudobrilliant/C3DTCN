#!/bin/bash

echo "-----Initializing Folder Structure-----"
rm -r ./temp
rm -r ./temp/truth
rm -r ./dataset
mkdir -p ./temp
mkdir -p ./temp/truth
mkdir -p ./dataset

echo "-----Retrieving Calibration-----"
wget -P ./temp http://4drepository.inrialpes.fr/data-4d/ixmas/setup.tgz
tar -zxvf ./temp/setup.tgz -C ./dataset
mv Calibration calibration

echo "-----Retrieving Ground Truth-----"
wget -P ./temp http://4drepository.inrialpes.fr/data-4d/ixmas/data/truth.txt.tgz
tar -zxvf ./temp/truth.txt.tgz -C ./temp/truth

echo "-----Retrieving Image Sets-----"
echo "Note: Depending on the number of img sets specified this may take some time."
declare -a img_paths=("julien1" "alba1" "alba2" "alba3" "amel1" "amel2" "amel3")

for i in "${img_paths[@]}"
do
    download_name="$i.pictures.tgz"
    wget -P ./temp "http://4drepository.inrialpes.fr/data-4d/ixmas/$i/$download_name"
    tar -zxvf ./temp/$download_name -C ./dataset
    img_truth=`find ./temp/truth -name "$i*"`
    cp ${img_truth} ./dataset/${i}_png/
done

rm ./temp -r

echo "-----Completed Dataset Creation-----"