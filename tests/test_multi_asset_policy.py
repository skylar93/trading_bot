#!/usr/bin/env python
"""Test script for multi-asset policy networks."""

import os
import sys
import unittest
import numpy as np
import torch
import pytest
from gymnasium.spaces import Box

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from networks.multi_asset_policy import MultiAssetLSTMPolicy, MultiAssetAttentionPolicy

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class TestMultiAssetLSTMPolicy(unittest.TestCase):
    """Test class for LSTM-based multi-asset policy network."""

    def setUp(self):
        """Set up the test environment."""
        self.n_assets = 3
        self.window_size = 10
        self.features_per_asset = 5
        self.total_features = self.n_assets * self.features_per_asset
        self.action_dim_per_asset = 1
        self.hidden_size = 64
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.total_features)
        )
        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_assets * self.action_dim_per_asset,)
        )
        
        # Create the policy network
        self.policy = MultiAssetLSTMPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_assets=self.n_assets,
            hidden_size=self.hidden_size,
            window_size=self.window_size,
            features_per_asset=self.features_per_asset,
            action_dim_per_asset=self.action_dim_per_asset
        )
        
        # Create sample input
        self.batch_size = 4
        self.sample_input = torch.randn(
            self.batch_size, 
            self.window_size, 
            self.total_features
        )
        
    def test_initialization(self):
        """Test proper initialization of the policy network."""
        # Check that network parameters exist
        self.assertIsNotNone(self.policy.lstm)
        self.assertIsNotNone(self.policy.shared_layers)
        self.assertEqual(len(self.policy.action_heads), self.n_assets)
        
        # Check attribute values
        self.assertEqual(self.policy.n_assets, self.n_assets)
        self.assertEqual(self.policy.window_size, self.window_size)
        self.assertEqual(self.policy.features_per_asset, self.features_per_asset)
        self.assertEqual(self.policy.total_features, self.total_features)
        self.assertEqual(self.policy.action_dim_per_asset, self.action_dim_per_asset)
    
    def test_forward_output_shape(self):
        """Test the output shape of the forward pass."""
        means, stds = self.policy.forward(self.sample_input)
        
        # Check output shapes
        expected_action_shape = (self.batch_size, self.n_assets * self.action_dim_per_asset)
        self.assertEqual(means.shape, expected_action_shape)
        self.assertEqual(stds.shape, expected_action_shape)
    
    def test_action_range(self):
        """Test that output actions are within the expected range."""
        actions = self.policy.get_action(self.sample_input, deterministic=True)
        
        # Actions should be in range [-1, 1] (tanh output)
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))
    
    def test_deterministic_vs_stochastic(self):
        """Test that deterministic and stochastic actions differ."""
        # Set seed for consistent results
        torch.manual_seed(42)
        det_actions1 = self.policy.get_action(self.sample_input, deterministic=True)
        
        # Reset seed to get same result
        torch.manual_seed(42)
        det_actions2 = self.policy.get_action(self.sample_input, deterministic=True)
        
        # Deterministic actions should be identical for the same input and seed
        self.assertTrue(torch.allclose(det_actions1, det_actions2, atol=1e-6))
        
        # Stochastic actions should differ
        # Reset seed for consistent test
        torch.manual_seed(42)
        stoch_actions1 = self.policy.get_action(self.sample_input, deterministic=False)
        
        # Use different seed for different result
        torch.manual_seed(43)
        stoch_actions2 = self.policy.get_action(self.sample_input, deterministic=False)
        
        # Stochastic actions should be different (with very high probability)
        # Instead of expecting exact inequality, check for significant difference
        diff = torch.abs(stoch_actions1 - stoch_actions2).mean()
        self.assertGreater(diff, 0.01)  # At least some significant difference
    
    def test_asset_specific_actions(self):
        """Test that each asset gets a dedicated action value."""
        # Use unique input per asset to clearly see per-asset output
        unique_input = self.sample_input.clone()
        
        # Create distinctive patterns for each asset
        for asset_idx in range(self.n_assets):
            start_idx = asset_idx * self.features_per_asset
            end_idx = start_idx + self.features_per_asset
            
            # Scale the input features for this asset to make them distinctive
            scale_factor = asset_idx + 1  # 1, 2, 3, ...
            unique_input[:, :, start_idx:end_idx] *= scale_factor
        
        # Get deterministic actions for clearer analysis
        actions = self.policy.get_action(unique_input, deterministic=True)
        
        # Check that actions for each asset are different
        for i in range(self.n_assets - 1):
            asset_i_action = actions[:, i * self.action_dim_per_asset:(i + 1) * self.action_dim_per_asset]
            asset_j_action = actions[:, (i + 1) * self.action_dim_per_asset:(i + 2) * self.action_dim_per_asset]
            
            # The actions should differ due to the distinctive inputs
            self.assertFalse(torch.allclose(asset_i_action, asset_j_action, atol=1e-3))


class TestMultiAssetAttentionPolicy(unittest.TestCase):
    """Test class for Attention-based multi-asset policy network."""

    def setUp(self):
        """Set up the test environment."""
        self.n_assets = 3
        self.window_size = 10
        self.features_per_asset = 5
        self.total_features = self.n_assets * self.features_per_asset
        self.action_dim_per_asset = 1
        self.hidden_size = 64
        self.num_heads = 2
        
        # Define observation and action spaces
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.total_features)
        )
        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_assets * self.action_dim_per_asset,)
        )
        
        # Create the policy network
        self.policy = MultiAssetAttentionPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            n_assets=self.n_assets,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            window_size=self.window_size,
            features_per_asset=self.features_per_asset,
            action_dim_per_asset=self.action_dim_per_asset
        )
        
        # Create sample input
        self.batch_size = 4
        self.sample_input = torch.randn(
            self.batch_size, 
            self.window_size, 
            self.total_features
        )
    
    def test_initialization(self):
        """Test proper initialization of the attention policy network."""
        # Check that network components exist
        self.assertIsNotNone(self.policy.feature_embedding)
        self.assertIsNotNone(self.policy.transformer_encoder)
        self.assertIsNotNone(self.policy.asset_attention)
        self.assertEqual(len(self.policy.action_heads), self.n_assets)
        
        # Check positional encoding
        self.assertIsNotNone(self.policy.positional_encoding)
        self.assertEqual(self.policy.positional_encoding.shape, (self.window_size, self.hidden_size))
        
        # Check attribute values
        self.assertEqual(self.policy.n_assets, self.n_assets)
        self.assertEqual(self.policy.window_size, self.window_size)
        self.assertEqual(self.policy.features_per_asset, self.features_per_asset)
    
    def test_forward_output_shape(self):
        """Test the output shape of the forward pass."""
        means, stds = self.policy.forward(self.sample_input)
        
        # Check output shapes
        expected_action_shape = (self.batch_size, self.n_assets * self.action_dim_per_asset)
        self.assertEqual(means.shape, expected_action_shape)
        self.assertEqual(stds.shape, expected_action_shape)
    
    def test_action_range(self):
        """Test that output actions are within the expected range."""
        actions = self.policy.get_action(self.sample_input, deterministic=True)
        
        # Actions should be in range [-1, 1] (tanh output)
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))
    
    def test_positional_encoding(self):
        """Test the positional encoding values."""
        pe = self.policy.positional_encoding
        
        # Test expected properties of positional encoding
        self.assertEqual(pe.shape, (self.window_size, self.hidden_size))
        
        # Check that different positions have different encodings
        for i in range(self.window_size - 1):
            self.assertFalse(torch.allclose(pe[i], pe[i+1]))
    
    def test_asset_interactions(self):
        """Test that attention mechanism can capture asset interactions."""
        # Fixing random seed for reproducibility
        torch.manual_seed(42)
        
        # Create inputs with strong correlations between assets
        correlated_input = torch.zeros(
            self.batch_size, 
            self.window_size, 
            self.total_features
        )
        
        # Create very clear correlation pattern between assets 0 and 1
        base_pattern = torch.randn(self.batch_size, self.window_size, self.features_per_asset)
        
        # Asset 0 follows the base pattern
        start0 = 0
        end0 = self.features_per_asset
        correlated_input[:, :, start0:end0] = base_pattern
        
        # Asset 1 follows almost the same pattern with minimal noise
        start1 = self.features_per_asset
        end1 = 2 * self.features_per_asset
        correlated_input[:, :, start1:end1] = base_pattern + torch.randn_like(base_pattern) * 0.05
        
        # Asset 2 has a completely different pattern
        start2 = 2 * self.features_per_asset
        end2 = 3 * self.features_per_asset
        correlated_input[:, :, start2:end2] = torch.randn(
            self.batch_size, self.window_size, self.features_per_asset
        )
        
        # Get actions for correlated input
        torch.manual_seed(42)  # Ensure reproducible output
        actions = self.policy.get_action(correlated_input, deterministic=True)
        
        # Split actions by asset
        asset0_actions = actions[:, 0:self.action_dim_per_asset]
        asset1_actions = actions[:, self.action_dim_per_asset:2*self.action_dim_per_asset]
        asset2_actions = actions[:, 2*self.action_dim_per_asset:3*self.action_dim_per_asset]
        
        # Manually calculate correlation between asset actions
        a0 = asset0_actions.squeeze().detach()
        a1 = asset1_actions.squeeze().detach()
        a2 = asset2_actions.squeeze().detach()
        
        # Calculate correlations manually to avoid issues with torch.corrcoef
        def correlation(x, y):
            x_mean = x.mean()
            y_mean = y.mean()
            x_std = x.std()
            y_std = y.std()
            
            if x_std == 0 or y_std == 0:
                return 0.0
                
            corr = ((x - x_mean) * (y - y_mean)).mean() / (x_std * y_std)
            return corr
        
        corr_01 = correlation(a0, a1)
        corr_02 = correlation(a0, a2)
        
        # Due to the very strong correlation in input, actions should be correlated
        # Since we specifically designed the input with strong correlation between 
        # assets 0 and 1, their actions should be more correlated than actions 
        # between 0 and 2
        self.assertGreater(abs(corr_01), abs(corr_02))


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 