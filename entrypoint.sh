#!/bin/bash

if [ "$(huggingface-cli whoami)" == "Not logged in" ]; then
        huggingface-cli login
fi

read -p "Enter your image idea string: " input
sed -i -e s/prompt=".*"/prompt="'${input}',"/g generate_images.py

echo "Proccessing... (this may take a while)"
/root/anaconda3/condabin/conda run -n sd_playground python generate_images.py