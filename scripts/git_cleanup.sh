#!/bin/bash

# Script to help clean up large files from Git history
# Use with caution - this rewrites Git history!

echo "This script will help you find and remove large files from your Git history."
echo "WARNING: This will rewrite your Git history. Make sure you understand the implications."
echo "It's recommended to backup your repository before proceeding."
echo ""
echo "Press Ctrl+C to cancel or Enter to continue..."
read

# Find the largest files in the Git repository
echo "Finding the largest files in your Git repository..."
git rev-list --objects --all | grep -v "^[0-9a-f]\{40\} " | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    sed -n 's/^blob //p' | sort -k2nr | head -25 | \
    awk '{print $3 " " $1 " " $2 " bytes"}' > large_files.txt

echo "Top 25 largest files in your Git repository:"
cat large_files.txt
echo ""

echo "Would you like to remove any of these files from your Git history? (y/n)"
read answer

if [ "$answer" != "y" ]; then
    echo "Exiting without making changes."
    exit 0
fi

echo "Enter the path of the file you want to remove (copy from the list above):"
read file_path

if [ -z "$file_path" ]; then
    echo "No file path provided. Exiting."
    exit 0
fi

echo "Removing $file_path from Git history..."
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch '$file_path'" --prune-empty --tag-name-filter cat -- --all

echo "File removed from Git history. You'll need to force push these changes."
echo "Run: git push origin --force --all"
echo ""
echo "Note: This only removes the file from Git history. If you want to keep the file"
echo "in your working directory but not track it, add it to .gitignore." 