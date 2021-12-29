#!/bin/bash

DIR="data/hover"

# if [ ! -d $DIR ]
# then 
#   echo "Directory $DIR does not exist exists. Creating directory"
#   mkdir $DIR
# else 
#   if [ "$(ls -A $DIR)" ]; then
#     echo "Directory $DIR not empty, skipping downloading HoVer dataset"
#     exit 1
#   fi
# fi

cd $DIR

# Datasets
if [ ! -f "train.json" ]; then
  echo "Downloading HoVer train set..."
  wget -O train.json https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_train_release_v1.1.json
else
  echo "train.json already exists, skipping download"
fi

if [ ! -f "dev.json" ]; then
  echo "Downloading HoVer dev set..."
  wget -O dev.json https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_dev_release_v1.1.json
else
  echo "dev.json already exists, skipping download"
fi

if [ ! -f "test.json" ]; then
  echo "Downloading HoVer test (claim only) set..."
  wget -O test.json https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_test_release_v1.1.json
else
  echo "test.json already exists, skipping download"
fi

if [ ! -f "sample_dev_pred.json" ]; then
  echo "Downloading HoVer sample dev predictions..."
  wget https://hover-nlp.github.io/data/hover/sample_dev_pred.json
else
  echo "sample_dev_pred.json already exists, skipping download"
fi

# Wiki dump
# wget -O wiki.tar.bz https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
# tar -xjvf wiki.tar.bz

# Wiki DB
if [ ! -f "wiki_wo_links.db" ]; then
  echo "Downloading Wiki dump database..."
  wget https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db
else
  echo "wiki_wo_links.db already exists, skipping download"
fi

echo "Finished downloading HoVer dataset into '$DIR'"

