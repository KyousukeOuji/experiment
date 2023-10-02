#!/bin/bash

for file in *.json; do
    labelme_json_to_dataset "$file"
done