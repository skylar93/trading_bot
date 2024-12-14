"""
Ray Tune based hyperparameter optimizer
"""
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import numpy as np
from typing import Dict, Optional
import logging
import mlflow

from .hyperopt_env import SimplifiedTradingEnv
from .hyperopt_agent import MinimalPPOAgent

logger = logging.getLogger(__name__)

def train_agent(env: SimplifiedTradingEnv, 
                agent: MinimalPPOAgent, 
                total_timesteps: int) -> Dict:
    """Training loop with early stopping"""
    state = env.reset()[0]
    best_reward = float('-inf')
    patience = 5
    no_improve = 0
    
    for step in range(total_timesteps):
        # Step environment
        action = agent.get_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        # Train agent
        metrics = agent.train(state, action, reward, next_state, done)
        
        # Early stopping check
        if reward > best_reward:
            best_reward = reward
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.debug(f"Early stopping at step {step}")
                break
        
        # Reset or update state
        if done or truncated:
            state = env.reset()[0]
        else:
            state = next_state
            
    return metrics

class MinimalTuner:
    def __init__(self, df, mlflow_experiment: Optional[str] = None):
        self.df = df
        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

    def objective(self, config: Dict) -> None:
        """Minimal objective function for Ray Tune"""
        # Create env and agent
        env = SimplifiedTradingEnv(
            df=self.df,
            initial_balance=config["initial_balance"],
            transaction_fee=config["transaction_fee"]
        )
        
        agent = MinimalPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            learning_rate=config["learning_rate"],
            hidden_size=config["hidden_size"],
            gamma=config["gamma"],
            epsilon=config["epsilon"]
        )
        
        # Train
        metrics = train_agent(env, agent, config["total_timesteps"])
        
        # Get final performance stats
        stats = env.get_episode_infos()[-1]  # Last episode stats
        
        # Report results
        tune.report(
            loss=metrics["loss"],
            sharpe_ratio=stats["sharpe_ratio"],
            total_return=stats["total_return"]
        )

    def run_optimization(self, num_samples: int = 10) -> Dict:
        """Run hyperparameter optimization"""
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-3),
            "hidden_size": tune.choice([32, 64]),
            "gamma": tune.uniform(0.95, 0.99),
            "epsilon": tune.uniform(0.1, 0.3),
            "initial_balance": tune.choice([10000]),
            "transaction_fee": tune.uniform(0.0005, 0.001),
            "total_timesteps": tune.choice([1000])  # Fixed for quick iteration
        }

        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='sharpe_ratio',
            mode='max',
            max_t=3,  # Maximum number of iterations
            grace_period=1
        )

        reporter = tune.CLIReporter(
            metric_columns=["loss", "sharpe_ratio", "total_return"]
        )

        analysis = tune.run(
            self.objective,
            num_samples=num_samples,
            config=search_space,
            scheduler=scheduler,
            progress_reporter=reporter,
            resources_per_trial={"cpu": 1},
            local_dir="./ray_results",
            name="minimal_trading_tune",
            verbose=0
        )
        
        best_config = analysis.get_best_config(metric="sharpe_ratio", mode="max")
        logger.info(f"Best config: {best_config}")
        
        return best_config

    def evaluate_config(self, config: Dict, episodes: int = 3) -> Dict:
        """Evaluate a configuration"""
        env = SimplifiedTradingEnv(
            df=self.df, 
            initial_balance=config["initial_balance"],
            transaction_fee=config["transaction_fee"]
        )
        
        agent = MinimalPPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            **{k: v for k, v in config.items() 
               if k in ["learning_rate", "hidden_size", "gamma", "epsilon"]}
        )
        
        # Run evaluation episodes
        metrics = []
        for _ in range(episodes):
            state = env.reset()[0]
            done = truncated = False
            
            while not (done or truncated):
                action = agent.get_action(state)
                state, _, done, truncated, _ = env.step(action)
            
            metrics.append(env.get_episode_infos()[-1])
        
        # Average metrics
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = float(np.mean([m[key] for m in metrics]))
        
        return avg_metrics