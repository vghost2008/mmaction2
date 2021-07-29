#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py boephone data/boephone/rawframes/ --level 2 --format rawframes --shuffle --num-split 1
echo "Filelist for rawframes generated."

cd tools/data/ucf101/
