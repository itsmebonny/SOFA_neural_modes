#!/bin/bash

find objects/ -type d -mindepth 1 -maxdepth 1 | while read -r subdir; do
  # Check if the subdirectory contains exactly one STL file
  num_stl_files=$(find "$subdir" -type f -name "*.stl" | wc -l)

  if [ "$num_stl_files" -eq 1 ]; then
    # Get the STL file
    stl_file=$(find "$subdir" -type f -name "*.stl")

    # Get the filename of the STL file
    stl_filename=$(basename "$stl_file")

    # Construct the destination path in the parent directory
    dest_path="objects/$stl_filename"

    # Move the STL file to the parent directory
    echo "Moving $stl_file to $dest_path"
    mv "$stl_file" "$dest_path"

    # Check if the move command was successful
    if [ $? -eq 0 ]; then
      echo "Successfully moved $stl_file to $dest_path"

      # Delete the now empty subdirectory
      echo "Deleting empty directory $subdir"
      rmdir "$subdir"

      # Check if the rmdir command was successful
      if [ $? -eq 0 ]; then
        echo "Successfully deleted $subdir"
      else
        echo "Error deleting $subdir"
      fi
    else
      echo "Error moving $stl_file"
    fi
  else
    echo "Subdirectory $subdir does not contain exactly one STL file. Skipping."
  fi
done

echo "Script finished"