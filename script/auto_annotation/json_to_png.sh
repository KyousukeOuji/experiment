#!/bin/bash

for file in ./binarized_data/out*.json; do
    labelme_json_to_dataset "{$file}"
done