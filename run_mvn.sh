#!/bin/bash

# Path to the CSV file

PROJECT_PATH=$1
CSV_FILE=$2

cd $PROJECT_PATH

# Check if the file exists
if [[ ! -f "$CSV_FILE" ]]; then
  echo "Error: File $CSV_FILE does not exist."
  exit 1
fi

# Read the CSV file line by line, skipping the header
while IFS="," read -r class_name method_signature; do
  # Skip the header row by checking for the "Class Name" string
  if [[ "$class_name" == "Class Name" && "$method_signature" == "Method Signature" ]]; then
    continue
  fi

  # Extract the method name (strip everything after and including the first '(')
  method_name=$(echo "$method_signature" | sed 's/(.*//')

  # Construct the mvn command
  command="mvn chatunitest:method -DselectMethod=${class_name}#${method_name}"

  # Print the command (for debugging or logging)
  echo "Running: $command"

  # Execute the command
  eval $command

done < "$CSV_FILE"
