#!/bin/bash

DIR="data/fever"

if [ ! -d $DIR ]
then 
  echo "Directory $DIR does not exist exists. Creating directory"
  mkdir $DIR
else 
  if [ "$(ls -A $DIR)" ]; then
    echo "Directory $DIR not empty, skipping downloading FEVER dataset"
    exit 1
  fi
fi

cd $DIR

# Dataset
wget https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
wget -O dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
wget -O test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl

# Wiki dump
wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
unzip wiki-pages
rm wiki-pages.zip
rm -rf __MACOSX/

echo "Finished downloading FEVER dataset into '$DIR'"

