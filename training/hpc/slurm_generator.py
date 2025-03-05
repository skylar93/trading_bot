"""
SLURM Job Script Generator for UW Hyak.

This module generates SLURM job scripts for running training jobs on UW Hyak
based on the unified configuration system. It supports both single-agent and 
multi-agent training with hyperparameter optimization.

Features:
- Generate SLURM job scripts from configuration
- Support for job arrays for hyperparameter sweeps
- GPU configuration for deep RL training
- Configuration of memory, CPU and time limits

Implementation Notes:
- Templates are used for different types of jobs
- Automatic selection of appropriate GPU/CPU resources
- Supports both single-job and array job submissions
- Files are saved with appropriate timestamps

Recent Changes:
- Added support for Ray cluster configuration
- Enhanced GPU memory management
- Added multi-node training support
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import sys

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.utils.config_manager import ConfigManager, load_config

logger = logging.getLogger(__name__)

# SLURM script templates
SLURM_HEADER_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --output={output}
{gpu_line}
{email_line}
{array_line}

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
{print_gpu_info}

# Load modules
module purge
module load anaconda3/2023.03
{additional_modules}

# Activate environment
source /sw/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate {conda_env}

# Set environment variables
export PYTHONPATH={project_root}:$PYTHONPATH
"""

SINGLE_JOB_TEMPLATE = """
cd {project_root}

# Run training script
python -u training/train_example.py \\
    --config {config_path} \\
    {multi_agent_flag} \\
    --experiment-id {experiment_id}
"""

RAY_CLUSTER_TEMPLATE = """
# Start Ray cluster
export RAY_memory={ray_head_memory}
export RAY_object_store_memory={ray_object_store_memory}

cd {project_root}

# Start Ray head node
python -u -m ray.cluster.entry_point --head \\
    --num-cpus={ray_num_cpus} \\
    --num-gpus={ray_num_gpus} \\
    --memory={ray_head_memory} \\
    --object-store-memory={ray_object_store_memory} \\
    --port=6379 \\
    --redis-password="{ray_redis_password}" \\
    --temp-dir={ray_temp_dir}

# Run Ray training script
python -u training/ray_train.py \\
    --config {config_path} \\
    --redis-address="$(hostname -i):6379" \\
    --redis-password="{ray_redis_password}" \\
    --experiment-id {experiment_id}
"""

HYPEROPT_ARRAY_TEMPLATE = """
cd {project_root}

# Extract hyperparameter set for this job array task
PARAM_FILE={param_files_dir}/params_${{SLURM_ARRAY_TASK_ID}}.yaml

# Run training with this parameter set
python -u training/hyperopt_train.py \\
    --base-config {config_path} \\
    --param-file $PARAM_FILE \\
    --experiment-id {experiment_id}_${{SLURM_ARRAY_TASK_ID}}
"""


def generate_slurm_script(
    config_path: str,
    output_dir: str = "training/hpc/jobs",
    experiment_id: Optional[str] = None,
    job_type: str = "single",  # 'single', 'ray', or 'hyperopt'
    conda_env: str = "trading_bot",
    additional_modules: List[str] = None
) -> str:
    """
    Generate a SLURM job script for UW Hyak based on the configuration.
    
    Args:
        config_path: Path to the configuration file
        output_dir: Directory to save the generated job script
        experiment_id: Optional experiment ID
        job_type: Type of job to generate (single, ray, hyperopt)
        conda_env: Conda environment name
        additional_modules: Additional modules to load
        
    Returns:
        Path to the generated job script
    """
    # Generate experiment ID if not provided
    if experiment_id is None:
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Load configuration
    config_manager = ConfigManager(default_config_path=config_path)
    config = config_manager.load_config()
    
    # Extract HPC configuration
    hpc_config = config.get("hpc", {})
    
    # Set default values if not specified
    scheduler = hpc_config.get("scheduler", "slurm")
    account = hpc_config.get("account", "your_account")
    partition = hpc_config.get("partition", "gpu-a40")
    nodes = hpc_config.get("nodes", 1)
    ntasks_per_node = hpc_config.get("ntasks_per_node", 1)
    cpus_per_task = hpc_config.get("cpus_per_task", 4)
    memory = hpc_config.get("memory", "48G")
    time_limit = hpc_config.get("time", "24:00:00")
    job_name = hpc_config.get("job_name", f"trading_bot_{experiment_id}")
    output_file = hpc_config.get("output", f"logs/slurm-%j.out")
    email = hpc_config.get("email", "")
    
    # GPU configuration
    gpus_per_node = hpc_config.get("gpus_per_node", 1)
    gpu_line = f"#SBATCH --gpus-per-node={gpus_per_node}" if gpus_per_node > 0 else ""
    print_gpu_info = "nvidia-smi" if gpus_per_node > 0 else "echo 'No GPU requested'"
    
    # Email notification
    email_line = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=ALL" if email else ""
    
    # Array job configuration
    array_size = None
    array_line = ""
    if job_type == "hyperopt":
        # For hyperopt, create a job array based on the number of hyperparameter samples
        hyperopt_config = config.get("hyperopt", {})
        array_size = hyperopt_config.get("num_samples", 10)
        array_line = f"#SBATCH --array=0-{array_size-1}"
    
    # Additional modules
    modules_str = "\n".join([f"module load {module}" for module in (additional_modules or [])])
    
    # Ray cluster configuration
    ray_config = {
        "ray_head_memory": hpc_config.get("ray_head_memory", "16G"),
        "ray_worker_memory": hpc_config.get("ray_worker_memory", "12G"),
        "ray_object_store_memory": hpc_config.get("ray_object_store_memory", "8G"),
        "ray_num_cpus": cpus_per_task,
        "ray_num_gpus": gpus_per_node,
        "ray_redis_password": "trading_bot_ray",
        "ray_temp_dir": f"/tmp/ray_{experiment_id}"
    }
    
    # Prepare the header
    header = SLURM_HEADER_TEMPLATE.format(
        job_name=job_name,
        account=account,
        partition=partition,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task,
        memory=memory,
        time=time_limit,
        output=output_file,
        gpu_line=gpu_line,
        email_line=email_line,
        array_line=array_line,
        print_gpu_info=print_gpu_info,
        additional_modules=modules_str,
        conda_env=conda_env,
        project_root=project_root
    )
    
    # Prepare the job-specific part
    if job_type == "single":
        # Single training job
        env_type = config["env"]["type"]
        multi_agent_flag = "--multi-agent" if env_type == "multi_agent_rl" else ""
        
        job_script = SINGLE_JOB_TEMPLATE.format(
            project_root=project_root,
            config_path=config_path,
            multi_agent_flag=multi_agent_flag,
            experiment_id=experiment_id
        )
    
    elif job_type == "ray":
        # Ray cluster job
        job_script = RAY_CLUSTER_TEMPLATE.format(
            project_root=project_root,
            config_path=config_path,
            experiment_id=experiment_id,
            **ray_config
        )
    
    elif job_type == "hyperopt":
        # Hyperparameter optimization with job array
        param_files_dir = os.path.join(project_root, "training/hpc/param_files", experiment_id)
        os.makedirs(param_files_dir, exist_ok=True)
        
        # Generate parameter files for each job in the array
        hyperopt_config = config["hyperopt"]
        params_space = hyperopt_config.get("parameters", {})
        
        # For this example, we'll just create dummy parameter files
        # In a real implementation, you'd generate systematic variations
        for i in range(array_size):
            param_file = os.path.join(param_files_dir, f"params_{i}.yaml")
            with open(param_file, "w") as f:
                yaml.dump({"task_id": i, "params": params_space}, f)
        
        job_script = HYPEROPT_ARRAY_TEMPLATE.format(
            project_root=project_root,
            config_path=config_path,
            param_files_dir=param_files_dir,
            experiment_id=experiment_id
        )
    
    else:
        raise ValueError(f"Unsupported job type: {job_type}")
    
    # Combine header and job script
    full_script = header + job_script
    
    # Save the job script
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"{job_type}_{experiment_id}_{timestamp}.sh"
    script_path = os.path.join(output_dir, script_filename)
    
    with open(script_path, "w") as f:
        f.write(full_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    logger.info(f"Generated SLURM job script: {script_path}")
    
    return script_path


def main():
    """Main function to run the SLURM job script generator."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM job scripts for UW Hyak from configuration"
    )
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="training/hpc/jobs",
        help="Directory to save job scripts"
    )
    parser.add_argument(
        "--experiment-id", type=str, default=None,
        help="Experiment ID"
    )
    parser.add_argument(
        "--job-type", type=str, choices=["single", "ray", "hyperopt"], default="single",
        help="Type of job to generate"
    )
    parser.add_argument(
        "--conda-env", type=str, default="trading_bot",
        help="Conda environment name"
    )
    parser.add_argument(
        "--modules", type=str, nargs="+", default=[],
        help="Additional modules to load"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate the job script
    script_path = generate_slurm_script(
        config_path=args.config,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        job_type=args.job_type,
        conda_env=args.conda_env,
        additional_modules=args.modules
    )
    
    print(f"Generated SLURM job script: {script_path}")
    print(f"To submit the job, run: sbatch {script_path}")


if __name__ == "__main__":
    main() 