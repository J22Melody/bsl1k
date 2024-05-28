#!/bin/bash

directory_in=$1
directory_out=$2

for file_in in $(find "$directory_in" -name '*.mpg' -or -name '*.mp4'); do
    file_out="${file_in//.mpg/.npy}"
    file_out=${file_out##*/}
    file_out="$directory_out$file_out"
    echo $file_in
    echo $file_out
    python demo_feature.py $file_in $file_out
done