class PolicyNetwork(nn.Module):
    """A simple MLP policy network for continuous action spaces.
    
    Features:
    - Outputs mean and standard deviation for Normal distribution
    - Uses tanh activation for bounded action space
    - Ensures positive standard deviation through softplus
    
    Implementation Notes:
    - Input shape: (batch_size, observation_size) or (observation_size,)
    - Output shape: (batch_size, action_size) for both mean and std
    - Uses 2 hidden layers with ReLU activation
    """
    
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        hidden_size: int = 256,
        activation: Type[nn.Module] = nn.ReLU,
        min_std: float = 1e-6,
        max_std: float = 1.0,
    ):
        """Initialize the policy network.
        
        Args:
            observation_space: Observation space (must be Box)
            action_space: Action space (must be Box)
            hidden_size: Size of hidden layers
            activation: Activation function to use
            min_std: Minimum standard deviation
            max_std: Maximum standard deviation
        """
        super().__init__()
        
        if not isinstance(observation_space, Box):
            raise ValueError("Observation space must be Box")
        if not isinstance(action_space, Box):
            raise ValueError("Action space must be Box")
            
        self.observation_size = np.prod(observation_space.shape)
        self.action_size = np.prod(action_space.shape)
        self.min_std = min_std
        self.max_std = max_std
        
        # Network layers
        self.net = nn.Sequential(
            nn.Linear(self.observation_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
        )
        
        # Output layers for mean and std
        self.mean_layer = nn.Linear(hidden_size, self.action_size)
        self.std_layer = nn.Linear(hidden_size, self.action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, observation_size) or (observation_size,)
            
        Returns:
            Tuple of (action_mean, action_std) tensors
        """
        # Ensure input is at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Check for NaN in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Main network
        features = self.net(x)
        
        # Mean with tanh to bound actions
        action_mean = torch.tanh(self.mean_layer(features))
        
        # Standard deviation with softplus and clipping
        action_std = F.softplus(self.std_layer(features))
        action_std = torch.clamp(action_std, min=self.min_std, max=self.max_std)
        
        return action_mean, action_std 