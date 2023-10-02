#!/bin/bash

for file in out*.json; do
    labelme_json_to_dataset "$file"
done