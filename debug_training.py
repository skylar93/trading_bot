"""
Debug Training Script for Single Asset, Single Agent Training.

This script simulates the training flow used by the UI but in a direct console script
to help debug training issues.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import yaml
import json
import torch
from typing import Dict, Any
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to ERROR to reduce output
logging.getLogger('agents.base.dummy_agent').setLevel(logging.ERROR)  # Reduce DummyAgent logs
logging.getLogger('DummyAgent').setLevel(logging.ERROR)  # Suppress DummyAgent logs
logging.getLogger('agents.strategies.single.dummy_agent').setLevel(logging.ERROR)  # Suppress DummyAgent logs
logging.getLogger('dummy_agent').setLevel(logging.ERROR)  # Suppress any other DummyAgent logs

logger = logging.getLogger(__name__)

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from data.utils.data_loader import DataLoader
from training.train_pipeline import train_pipeline
from agents.strategies.agent_factory import create_agent
from training.env_factory import create_env
from training.utils.unified_mlflow_manager import MLflowManager

def load_data(symbol="BTC/USDT", timeframe="1h", start_date="2022-01-01", end_date="2022-01-10"):
    """
    Load historical data for the given trading pair
    """
    try:
        # Log the loading process
        logging.info(f"📊 Loading data for {symbol} from {start_date} to {end_date}")
        
        # Create dataloader
        data_loader = DataLoader(
            exchange_id="binance",
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Fetch data - use a larger date range to ensure we have enough data
        df = data_loader.fetch_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Log the shape of the data
        logging.info(f"📈 Loaded data with shape: {df.shape}")
        
        # Display sample of the data
        logging.info("Sample data (first 3 rows):")
        logging.info(df.head(3))
        
        return df
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        raise

def create_training_config(
    symbol="BTC/USDT",
    timeframe="1h",
    agent_type="ppo",
    total_timesteps=5000,
    learning_rate=3e-4
):
    """
    Create a training configuration for the debug script
    """
    # Create a progress callback with enhanced metrics tracking
    def progress_callback(episode, step, recent_reward, losses=None):
        # Track metrics every episode to more closely monitor learning progress
        if losses:
            loss_str = f", Losses: {', '.join([f'{k}={v:.4f}' for k, v in losses.items()])}"
            
            # Create more detailed tracking for PPO-specific metrics
            ppo_metrics = []
            if 'policy_loss' in losses:
                ppo_metrics.append(f"Policy Loss: {losses['policy_loss']:.4f}")
            if 'value_loss' in losses:
                ppo_metrics.append(f"Value Loss: {losses['value_loss']:.4f}")
            if 'entropy' in losses:
                ppo_metrics.append(f"Entropy: {losses['entropy']:.4f}")
            if 'kl' in losses:
                ppo_metrics.append(f"KL Divergence: {losses['kl']:.4f}")
                
            if ppo_metrics:
                logger.info(f"PPO Metrics - Episode {episode}: {' | '.join(ppo_metrics)}")
        
        # Return a message every 10 episodes for the standard progress log
        if episode % 10 == 0:
            loss_str = ""
            if losses:
                loss_str = f", Losses: {', '.join([f'{k}={v:.4f}' for k, v in losses.items()])}"
            return f"Episode {episode}, Step {step}, Recent Reward: {recent_reward:.2f}{loss_str}"
        return None
    
    # Output directory for checkpoints
    checkpoint_dir = "debug_checkpoints"
    
    # Generate a unique experiment name
    experiment_name = f"{agent_type}_{symbol}_{timeframe}"
    
    # Log the configuration
    logging.info(f"🚀 Starting debug training with parameters:")
    logging.info(f"Symbol: {symbol}")
    logging.info(f"Timeframe: {timeframe}")
    logging.info(f"Date Range: 2022-01-01 to 2022-01-10 (10 days)")
    logging.info(f"Agent Type: {agent_type}")
    logging.info(f"Learning Rate: {learning_rate}")
    logging.info(f"Total Timesteps: {total_timesteps}")
    
    # Return the full configuration
    return {
        "data": {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": "2022-01-01",
            "end_date": "2022-01-10",  # Use more days to have enough data
            "test_size": 0.2,
            "random_state": 42
        },
        "env": {
            "type": "single_asset_rl",
            "window_size": 60,
            "initial_capital": 10000,
            "trading_fee": 0.001,
            "risk_adjusted_reward": True,
            "apply_slippage": True
        },
        "agent": {
            "type": agent_type,
            "learning_rate": learning_rate,
            "gamma": 0.99,
            "normalize_observations": True,
            "track_gradients": True,
            "log_weight_histograms": True
        },
        "training": {
            "total_timesteps": total_timesteps,
            "batch_size": 64,
            "progress_callback": progress_callback,
            "save_interval": 2000,
            "log_interval": 1,
            "eval_interval": 1000
        },
        # Moved checkpoint_dir to paths section to match train_pipeline.py structure
        "paths": {
            "checkpoint_dir": checkpoint_dir
        },
        "mlflow_manager": {
            "tracking_dir": "./mlruns",
            "experiment_name": experiment_name,
            "run_name": f"{experiment_name}_{int(time.time())}",
            "log_artifacts": True
        }
    }

def analyze_learning_progress(results, checkpoint_dir):
    """
    Analyze training results to determine if learning is occurring
    
    Args:
        results: Dictionary of training results
        checkpoint_dir: Directory where model checkpoints are saved
    """
    logger.info("\n===== LEARNING ANALYSIS =====")
    
    # Check if we have episode rewards
    if 'episode_rewards' in results:
        rewards = results['episode_rewards']
        num_episodes = len(rewards)
        
        if num_episodes > 5:  # Need enough episodes to analyze trends
            # Calculate statistics
            mean_reward = np.mean(rewards)
            median_reward = np.median(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            
            # Calculate reward trends (using chunks to smooth noise)
            chunk_size = max(1, num_episodes // 5)
            reward_chunks = [rewards[i:i+chunk_size] for i in range(0, num_episodes, chunk_size)]
            chunk_means = [np.mean(chunk) for chunk in reward_chunks if len(chunk) > 0]
            
            logger.info(f"Reward Statistics:")
            logger.info(f"  Mean Reward: {mean_reward:.2f}")
            logger.info(f"  Median Reward: {median_reward:.2f}")
            logger.info(f"  Min/Max Reward: {min_reward:.2f}/{max_reward:.2f}")
            
            # Check for reward improvement trend
            if len(chunk_means) >= 3:
                first_chunk = chunk_means[0]
                last_chunk = chunk_means[-1]
                logger.info(f"  Reward Trend (first chunk → last chunk): {first_chunk:.2f} → {last_chunk:.2f}")
                
                if last_chunk > first_chunk * 1.1:  # 10% improvement
                    logger.info("  ✅ POSITIVE TREND: Rewards are improving significantly")
                elif last_chunk > first_chunk:
                    logger.info("  ⚠️ WEAK IMPROVEMENT: Rewards are slightly improving")
                elif last_chunk < first_chunk * 0.9:  # 10% decline
                    logger.info("  ❌ NEGATIVE TREND: Rewards are declining significantly")
                else:
                    logger.info("  ⚠️ NO CLEAR TREND: Rewards are relatively stable")
            else:
                logger.info("  ⚠️ Not enough data to analyze reward trends")
    
    # Check model files for size/content consistency
    best_model_path = os.path.join(checkpoint_dir, "best_agent.pt")
    final_model_path = os.path.join(checkpoint_dir, "final_agent.pt")
    
    if os.path.exists(best_model_path) and os.path.exists(final_model_path):
        best_size = os.path.getsize(best_model_path)
        final_size = os.path.getsize(final_model_path)
        
        logger.info(f"Model File Sizes:")
        logger.info(f"  Best Model: {best_size/1024:.1f} KB")
        logger.info(f"  Final Model: {final_size/1024:.1f} KB")
        
        if abs(best_size - final_size) < 100:  # Almost identical
            logger.info("  ⚠️ WARNING: Best and final models are almost identical in size")
        else:
            logger.info("  ✅ Models differ in size, suggesting learning occurred")
    else:
        logger.info("  ❌ One or both model files are missing")
    
    # Analyze episode lengths for exploration vs exploitation
    if 'episode_lengths' in results:
        lengths = results['episode_lengths']
        mean_length = np.mean(lengths)
        logger.info(f"Episode Length Analysis:")
        logger.info(f"  Mean Episode Length: {mean_length:.1f}")
        
        # Check consistency of episode lengths
        if np.std(lengths) < 0.01:
            logger.info("  ⚠️ WARNING: Episode lengths are extremely consistent, suggesting no exploration")
        else:
            logger.info("  ✅ Episode lengths vary, suggesting exploration is happening")
    
    logger.info("===== END ANALYSIS =====\n")
    
    # Return assessment of learning
    if 'episode_rewards' in results and len(chunk_means) >= 3:
        if last_chunk > first_chunk:
            return "LEARNING_DETECTED"
        else:
            return "NO_LEARNING_DETECTED"
    else:
        return "INSUFFICIENT_DATA"

def create_learning_visualizations(results, checkpoint_dir):
    """
    Create visualizations of learning progress
    
    Args:
        results: Dictionary of training results
        checkpoint_dir: Directory where model checkpoints are saved
    """
    # Ensure visualization directory exists
    vis_dir = os.path.join(checkpoint_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot episode rewards
    if 'episode_rewards' in results:
        rewards = results['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        # Add a trend line
        if len(rewards) > 2:
            z = np.polyfit(episodes, rewards, 1)
            p = np.poly1d(z)
            plt.plot(episodes, p(episodes), "r--", 
                     label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")
        
        # Add mean line
        mean_reward = np.mean(rewards)
        plt.axhline(y=mean_reward, color='g', linestyle='-', 
                   label=f'Mean: {mean_reward:.2f}')
        
        plt.legend()
        plt.savefig(os.path.join(vis_dir, 'episode_rewards.png'))
        plt.close()
        
        logger.info(f"Saved reward visualization to {vis_dir}/episode_rewards.png")
        
    # Plot reward moving average for clearer trend
    if 'episode_rewards' in results and len(results['episode_rewards']) > 5:
        rewards = results['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        
        window_size = min(10, len(rewards) // 3)
        if window_size > 0:
            moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) 
                          for i in range(len(rewards))]
            
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, rewards, alpha=0.3, label='Rewards')
            plt.plot(episodes, moving_avg, 'r', label=f'{window_size}-Episode Moving Avg')
            plt.title(f'Reward Trend Analysis (Window Size: {window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(vis_dir, 'reward_trend.png'))
            plt.close()
            
            logger.info(f"Saved reward trend visualization to {vis_dir}/reward_trend.png")
    
    # If we have loss data, plot it
    if hasattr(results.get('agent', {}), 'training_metrics'):
        metrics = results['agent'].training_metrics
        
        for metric_name, values in metrics.items():
            if len(values) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(values)
                plt.title(f'{metric_name} Over Training')
                plt.xlabel('Update')
                plt.ylabel(metric_name)
                plt.grid(True)
                plt.savefig(os.path.join(vis_dir, f'{metric_name.lower().replace(" ", "_")}.png'))
                plt.close()
                
        logger.info(f"Saved {len(metrics)} metric visualizations to {vis_dir}")
    
    return vis_dir

def analyze_model_weights(checkpoint_dir):
    """
    Analyze the model weights from saved checkpoints to determine if learning occurred
    
    Args:
        checkpoint_dir: Directory where model checkpoints are saved
        
    Returns:
        Dictionary with weight analysis results
    """
    best_model_path = os.path.join(checkpoint_dir, "best_agent.pt")
    final_model_path = os.path.join(checkpoint_dir, "final_agent.pt")
    
    if not os.path.exists(best_model_path) or not os.path.exists(final_model_path):
        logger.warning("Cannot analyze model weights: one or both model files missing")
        return {"status": "MISSING_FILES"}
    
    try:
        # Load the models
        best_model = torch.load(best_model_path, map_location=torch.device('cpu'))
        final_model = torch.load(final_model_path, map_location=torch.device('cpu'))
        
        logger.info("\n===== MODEL WEIGHT ANALYSIS =====")
        
        # Track differences between models
        total_params = 0
        diff_count = 0
        max_diff = 0
        max_diff_layer = ""
        
        analysis = {
            "total_params": 0,
            "changed_params": 0,
            "change_percentage": 0,
            "max_diff": 0,
            "max_diff_layer": "",
            "layer_diffs": {}
        }
        
        # Check policy network weights
        if "policy_state_dict" in best_model and "policy_state_dict" in final_model:
            best_policy = best_model["policy_state_dict"]
            final_policy = final_model["policy_state_dict"]
            
            # Compare each layer
            for param_name in best_policy:
                if param_name in final_policy:
                    best_weights = best_policy[param_name].numpy()
                    final_weights = final_policy[param_name].numpy()
                    
                    # Calculate differences
                    param_diff = np.abs(final_weights - best_weights)
                    diff_magnitude = np.mean(param_diff)
                    max_param_diff = np.max(param_diff)
                    
                    total_params += best_weights.size
                    diff_count += np.sum(param_diff > 1e-6)  # Count params that changed
                    
                    # Track max difference
                    if max_param_diff > max_diff:
                        max_diff = max_param_diff
                        max_diff_layer = param_name
                    
                    # Store layer-specific diff
                    analysis["layer_diffs"][param_name] = {
                        "mean_diff": float(diff_magnitude),
                        "max_diff": float(max_param_diff),
                        "changed_percentage": float(np.sum(param_diff > 1e-6) / best_weights.size * 100)
                    }
                    
                    # Log significant differences
                    if diff_magnitude > 0.01:
                        logger.info(f"Layer {param_name}: Mean diff {diff_magnitude:.6f}, Max diff {max_param_diff:.6f}")
        
        # Check value network weights too
        if "value_state_dict" in best_model and "value_state_dict" in final_model:
            best_value = best_model["value_state_dict"]
            final_value = final_model["value_state_dict"]
            
            # Compare each layer
            for param_name in best_value:
                if param_name in final_value:
                    best_weights = best_value[param_name].numpy()
                    final_weights = final_value[param_name].numpy()
                    
                    # Calculate differences
                    param_diff = np.abs(final_weights - best_weights)
                    diff_magnitude = np.mean(param_diff)
                    max_param_diff = np.max(param_diff)
                    
                    total_params += best_weights.size
                    diff_count += np.sum(param_diff > 1e-6)  # Count params that changed
                    
                    # Track max difference
                    if max_param_diff > max_diff:
                        max_diff = max_param_diff
                        max_diff_layer = param_name
                    
                    # Store layer-specific diff
                    analysis["layer_diffs"][f"value_{param_name}"] = {
                        "mean_diff": float(diff_magnitude),
                        "max_diff": float(max_param_diff),
                        "changed_percentage": float(np.sum(param_diff > 1e-6) / best_weights.size * 100)
                    }
                    
                    # Log significant differences
                    if diff_magnitude > 0.01:
                        logger.info(f"Value Layer {param_name}: Mean diff {diff_magnitude:.6f}, Max diff {max_param_diff:.6f}")
        
        # Calculate overall statistics
        if total_params > 0:
            change_percentage = (diff_count / total_params) * 100
            analysis["total_params"] = int(total_params)
            analysis["changed_params"] = int(diff_count)
            analysis["change_percentage"] = float(change_percentage)
            analysis["max_diff"] = float(max_diff)
            analysis["max_diff_layer"] = max_diff_layer
            
            logger.info(f"Total parameters: {total_params}")
            logger.info(f"Changed parameters: {diff_count} ({change_percentage:.2f}%)")
            logger.info(f"Maximum difference: {max_diff:.6f} in layer {max_diff_layer}")
            
            # Determine if learning occurred based on weight changes
            if change_percentage > 10 and max_diff > 0.01:
                logger.info("✅ SIGNIFICANT LEARNING: More than 10% of weights changed substantially")
                analysis["status"] = "SIGNIFICANT_LEARNING"
            elif change_percentage > 1 and max_diff > 0.001:
                logger.info("⚠️ MODERATE LEARNING: Some weight changes detected")
                analysis["status"] = "MODERATE_LEARNING"
            else:
                logger.info("❌ MINIMAL LEARNING: Very few weight changes detected")
                analysis["status"] = "MINIMAL_LEARNING"
        else:
            logger.info("❌ Unable to analyze parameters: no valid parameters found")
            analysis["status"] = "INVALID_PARAMETERS"
        
        logger.info("===== END MODEL ANALYSIS =====\n")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing model weights: {e}")
        return {"status": "ERROR", "message": str(e)}

def main():
    # Define parameters for training
    symbol = "BTC/USDT"
    timeframe = "1h"
    agent_type = "ppo"
    learning_rate = 3e-4
    total_timesteps = 10000  # Increased from 5000 to 10000 for better learning observation
    
    # Log initial parameters
    logger.info(f"🚀 Starting debug training with parameters:")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Date Range: 2022-01-01 to 2022-01-10 (10 days)")
    logger.info(f"Agent Type: {agent_type}")
    logger.info(f"Learning Rate: {learning_rate}")
    logger.info(f"Total Timesteps: {total_timesteps}")
    
    # Load historical data
    data = load_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date="2022-01-01",
        end_date="2022-01-10"
    )
    
    # Create the training configuration
    config = create_training_config(
        symbol=symbol,
        timeframe=timeframe,
        agent_type=agent_type,
        learning_rate=learning_rate,
        total_timesteps=total_timesteps
    )
    
    # Add additional debugging options
    config["agent"]["track_gradients"] = True  # Track network gradients for debugging
    config["agent"]["log_weight_histograms"] = True  # Log weight histograms to MLflow
    config["training"]["log_interval"] = 1  # Log metrics every episode
    config["training"]["eval_interval"] = 1000  # Evaluate more frequently
    
    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.abspath(os.path.join(project_root, config["paths"]["checkpoint_dir"]))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Update the config with the absolute path
    config["paths"]["checkpoint_dir"] = checkpoint_dir
    
    # Create MLflowManager
    mlflow_cfg = config["mlflow_manager"]
    mlflow_manager = MLflowManager(
        experiment_name=mlflow_cfg["experiment_name"],
        tracking_dir=mlflow_cfg["tracking_dir"]
    )
    mlflow_manager.start_run(run_name=mlflow_cfg["run_name"])
    
    # Update config with the actual MLflowManager object
    config["mlflow_manager"] = mlflow_manager
    
    # Set up logging for specific modules
    # Set specific loggers to ERROR to reduce output
    logging.getLogger('agents.base.dummy_agent').setLevel(logging.ERROR)  # Reduce DummyAgent logs
    
    # Start the training pipeline
    logger.info("▶️ Starting training pipeline")
    try:
        results = train_pipeline(config=config, data=data)
        logger.info("✅ Training completed successfully!")
        logger.info(f"Results: {results}")
        
        # Check for model files
        best_model_path = os.path.join(checkpoint_dir, "best_agent.pt")
        final_model_path = os.path.join(checkpoint_dir, "final_agent.pt")
        
        logger.info(f"Checking for model files in {checkpoint_dir}:")
        logger.info(f"Best model exists: {os.path.exists(best_model_path)}")
        logger.info(f"Final model exists: {os.path.exists(final_model_path)}")
        
        # List all files in the checkpoint directory
        logger.info(f"Files in {checkpoint_dir}:")
        for file in os.listdir(checkpoint_dir):
            logger.info(f"  - {file}")
        
        # Analyze training results to determine if learning is occurring
        learning_assessment = analyze_learning_progress(results, checkpoint_dir)
        logger.info(f"Learning Assessment: {learning_assessment}")
        
        # Create visualizations of learning progress
        vis_dir = create_learning_visualizations(results, checkpoint_dir)
        logger.info(f"Learning visualizations saved to {vis_dir}")
        
        # Analyze model weights to determine if learning occurred
        weight_analysis = analyze_model_weights(checkpoint_dir)
        logger.info(f"Model Weight Analysis: {weight_analysis['status']}")
            
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        raise
    finally:
        # End the MLflow run
        if mlflow_manager:
            mlflow_manager.end_run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")