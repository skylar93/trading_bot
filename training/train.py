    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, callback=None):
        """Train the agent"""
        logger.info("Starting training...")
        
        # Create environments
        train_env = self.create_env(train_data)
        val_env = self.create_env(val_data)
        
        # Create agent
        agent = PPOAgent(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            hidden_dim=self.config['model']['fcnet_hiddens'][0],
            lr=self.config['model']['lr']
        )
        
        # Start MLflow run
        self.mlflow_manager.start_run()
        
        # Log configurations
        self.mlflow_manager.log_params(self.config['model'])
        self.mlflow_manager.log_params(self.config['env'])
        
        best_val_return = float('-inf')
        no_improve_count = 0
        
        for episode in range(self.config['training']['total_timesteps']):
            logger.info(f"Episode {episode+1}/{self.config['training']['total_timesteps']}")
            
            # Training episode
            train_metrics = self._run_episode(train_env, agent, train=True)
            logger.info(f"Train metrics: {train_metrics}")
            
            # Validation episode
            val_metrics = self._run_episode(val_env, agent, train=False)
            logger.info(f"Val metrics: {val_metrics}")
            
            # Call callback if provided
            if callback:
                callback(episode, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['episode_reward'] > best_val_return:
                best_val_return = val_metrics['episode_reward']
                self.save_model(agent, "best_model.pt")
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= self.config['training'].get('early_stop', 20):
                logger.info("Early stopping triggered")
                break
        
        # End MLflow run
        self.mlflow_manager.end_run()
        return agent