#!/bin/bash

DATASET_SLUG="hasibullahaman/traffic-prediction-dataset"

mkdir -p data

cd data

kaggle datasets download $DATASET_SLUG

unzip "${DATASET_SLUG##*/}.zip"
rm "${DATASET_SLUG##*/}.zip"

echo "Dataset downloaded and extracted to 'data/' directory."
