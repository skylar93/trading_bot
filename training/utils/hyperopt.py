class MinimalTuner:
    def __init__(self, config):
        self.config = config
        self.best_params = None
        self.best_score = float('-inf')
    
    def optimize(self, train_fn, search_space, n_trials=10):
        """Run hyperparameter optimization
        
        Args:
            train_fn: Training function to optimize
            search_space: Dictionary of parameter ranges to search
            n_trials: Number of trials to run
            
        Returns:
            Best parameters found
        """
        for _ in range(n_trials):
            # Sample parameters from search space
            params = {
                k: np.random.uniform(v[0], v[1]) 
                for k, v in search_space.items()
            }
            
            # Evaluate parameters
            score = train_fn(params)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        return self.best_params 