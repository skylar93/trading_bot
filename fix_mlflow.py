"""
Fix MLflow experiment ID inconsistencies.

This script fixes the mixed string/integer experiment ID issue in MLflow
by ensuring all experiment IDs are consistently stored as strings.
"""

import os
import sys
import shutil
import yaml
import re
from pathlib import Path

MLRUNS_DIR = "./mlruns"
BACKUP_DIR = "./mlruns_backup"
NEW_DIR = "./mlruns_fixed"

def main():
    print(f"Creating new MLflow directory structure in {NEW_DIR}")
    # Create new directory
    if os.path.exists(NEW_DIR):
        print(f"Removing existing {NEW_DIR}")
        shutil.rmtree(NEW_DIR)
    os.makedirs(NEW_DIR)
    
    # Copy only essential structure
    for item in os.listdir(BACKUP_DIR):
        src_path = os.path.join(BACKUP_DIR, item)
        
        # Skip any hidden files
        if item.startswith('.'):
            continue
            
        # Handle special directories
        if item in ['artifacts', 'models', '0']:
            print(f"Copying special directory: {item}")
            shutil.copytree(src_path, os.path.join(NEW_DIR, item))
            continue
        
        # For experiment directories, ensure ID is a string in meta.yaml
        if os.path.isdir(src_path) and re.match(r'^\d+$', item):
            meta_file = os.path.join(src_path, "meta.yaml")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r') as f:
                        meta_data = yaml.safe_load(f)
                    
                    # Ensure experiment_id is a string
                    if 'experiment_id' in meta_data:
                        meta_data['experiment_id'] = str(meta_data['experiment_id'])
                    
                    # Create destination directory
                    dst_path = os.path.join(NEW_DIR, item)
                    os.makedirs(dst_path, exist_ok=True)
                    
                    # Write fixed meta.yaml
                    with open(os.path.join(dst_path, "meta.yaml"), 'w') as f:
                        yaml.safe_dump(meta_data, f)
                    
                    # Copy all run directories
                    for run_item in os.listdir(src_path):
                        if run_item != "meta.yaml" and not run_item.startswith('.'):
                            run_src = os.path.join(src_path, run_item)
                            run_dst = os.path.join(dst_path, run_item)
                            if os.path.isdir(run_src):
                                shutil.copytree(run_src, run_dst)
                    
                    print(f"Fixed experiment directory: {item}")
                except Exception as e:
                    print(f"Error processing {meta_file}: {e}")
            else:
                # Just copy the directory as is
                shutil.copytree(src_path, os.path.join(NEW_DIR, item))
        else:
            # For other files, just copy them
            if os.path.isdir(src_path):
                shutil.copytree(src_path, os.path.join(NEW_DIR, item))
            else:
                shutil.copy2(src_path, os.path.join(NEW_DIR, item))
    
    print("\nFix completed. New MLflow directory structure in:", NEW_DIR)
    print("\nTo use the fixed directory:")
    print("1. Rename the current mlruns directory: mv mlruns mlruns_old")
    print("2. Rename the fixed directory: mv mlruns_fixed mlruns")
    print("3. Try running MLflow UI again: mlflow ui")

if __name__ == "__main__":
    main() 