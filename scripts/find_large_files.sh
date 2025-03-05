#!/bin/bash

# Script to find large files in the current directory
# This can help identify files that should be added to .gitignore

echo "Finding files larger than 10MB in the current directory..."
find . -type f -size +10M | grep -v ".git/" | sort -h

echo -e "\nFinding files larger than 1MB in the current directory..."
find . -type f -size +1M -size -10M | grep -v ".git/" | sort -h

echo -e "\nDo you want to see the total size of each directory? (y/n)"
read answer

if [ "$answer" = "y" ]; then
    echo -e "\nCalculating directory sizes..."
    du -h -d 1 | sort -hr
fi 