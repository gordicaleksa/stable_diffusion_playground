#!/bin/bash

if [ -f firstrun ] ; then
        docker start sd_playground || echo "Fixing error... please try again! $(rm firstrun) " 
        docker exec -it sd_playground bash entrypoint.sh
else
        docker run --name=sd_playground  -it --gpus=all -v ${PWD}/output:/output sd_playground
        touch firstrun
fi

cp -rp output $output_path

echo "Images can be found at: ${output_path}"