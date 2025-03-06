#!/bin/bash

find objects/ -type f -name "*.obj" | while read -r obj_file; do
  # Get the directory containing the obj file
  obj_dir=$(dirname "$obj_file")

  # Get the filename without the extension
  filename=$(basename "$obj_file" .obj)

  # Construct the output STL filename
  stl_file="$obj_dir/${filename}.stl"

  # Execute the assimp command
  echo "Converting $obj_file to $stl_file"
  assimp export "$obj_file" "$stl_file"

  # Check if the assimp command was successful
  if [ $? -eq 0 ]; then
    echo "Successfully converted $obj_file to $stl_file"

    # Delete all files in the directory except the newly created STL file
    find "$obj_dir/" -type f ! -name "$(basename "$stl_file")" -delete
    echo "Deleted all files except $stl_file in $obj_dir"
  else
    echo "Error converting $obj_file"
  fi
done

echo "Script finished"