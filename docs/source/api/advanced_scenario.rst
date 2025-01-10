Advanced Scenario API
====================

Overview
--------

The advanced scenario module provides tools for generating and testing complex market scenarios such as flash crashes, low liquidity periods, and choppy markets.

Class Structure
--------------

ScenarioGenerator
^^^^^^^^^^^^^^^

.. code-block:: python

    class ScenarioGenerator:
        """Generator for advanced market scenarios."""
        
        def __init__(self, base_price=100, volatility=0.02):
            """Initialize scenario generator.
            
            Args:
                base_price: Starting price level
                volatility: Base volatility level
            """
            self.base_price = base_price
            self.volatility = volatility

Key Methods
----------

Flash Crash Generation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def generate_flash_crash(self, n_steps=1000, crash_idx=500,
                           crash_size=0.3, recovery_ratio=0.5):
        """Generate flash crash scenario.
        
        Args:
            n_steps: Total number of steps
            crash_idx: When crash occurs
            crash_size: Size of price drop
            recovery_ratio: How much price recovers
            
        Returns:
            DataFrame: OHLCV data with flash crash
        """
        # Generate base price series
        prices = self._generate_base_prices(n_steps)
        
        # Add crash and recovery
        crash_point = prices[crash_idx]
        prices[crash_idx:] *= (1 - crash_size)
        recovery = crash_size * recovery_ratio
        prices[crash_idx+1:] *= (1 + recovery)
        
        return self._to_ohlcv(prices)

Low Liquidity Generation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def generate_low_liquidity(self, n_steps=1000,
                             low_liq_start=300,
                             low_liq_duration=200):
        """Generate low liquidity scenario.
        
        Args:
            n_steps: Total number of steps
            low_liq_start: Start of low liquidity
            low_liq_duration: Duration of low liquidity
            
        Returns:
            DataFrame: OHLCV data with low liquidity period
        """
        # Generate base scenario
        df = self._generate_base_scenario(n_steps)
        
        # Modify volatility and volume
        low_liq_end = low_liq_start + low_liq_duration
        df.loc[low_liq_start:low_liq_end, 'volume'] *= 0.2
        df.loc[low_liq_start:low_liq_end, ['high', 'low']] *= 1.5
        
        return df

Scenario Testing
^^^^^^^^^^^^^^^

.. code-block:: python

    class ScenarioTester:
        """Tester for advanced market scenarios."""
        
        def __init__(self, env, agent):
            """Initialize scenario tester.
            
            Args:
                env: Trading environment
                agent: Trading agent
            """
            self.env = env
            self.agent = agent
            
        def test_flash_crash(self):
            """Test agent behavior during flash crash.
            
            Returns:
                dict: Performance metrics during crash
            """
            metrics = {
                'survival': True,
                'max_drawdown': 0.0,
                'recovery_time': 0,
                'final_value': 0.0
            }
            
            # Run simulation
            state = self.env.reset()
            for step in range(1000):
                action = self.agent.get_action(state)
                state, reward, done, info = self.env.step(action)
                
                # Update metrics
                metrics['max_drawdown'] = max(
                    metrics['max_drawdown'],
                    info['drawdown']
                )
                if done:
                    metrics['final_value'] = info['portfolio_value']
                    break
                    
            return metrics

Implementation Details
--------------------

Scenario Generation
^^^^^^^^^^^^^^^^^

1. Base Price Generation:
   - Random walk with drift
   - Controlled volatility
   - Optional seasonality

2. Event Injection:
   - Flash crashes
   - Volatility spikes
   - Volume changes

3. OHLCV Construction:
   - Price to candlestick conversion
   - Volume profile generation
   - Data validation

Dependencies
-----------

- NumPy (random generation)
- Pandas (DataFrame operations)
- TradingEnvironment
- PPOAgent

Usage Example
------------

Basic Usage
^^^^^^^^^^

.. code-block:: python

    # Create generator
    generator = ScenarioGenerator(base_price=100)
    
    # Generate scenarios
    flash_crash_data = generator.generate_flash_crash(
        n_steps=1000,
        crash_size=0.3
    )
    
    low_liq_data = generator.generate_low_liquidity(
        n_steps=1000,
        low_liq_duration=200
    )
    
    # Test scenarios
    tester = ScenarioTester(env, agent)
    flash_crash_metrics = tester.test_flash_crash()
    print(f"Max Drawdown: {flash_crash_metrics['max_drawdown']}")

Best Practices
------------

1. Scenario Design
^^^^^^^^^^^^^^^

- Use realistic parameters
- Validate generated data
- Consider market mechanics

2. Testing Strategy
^^^^^^^^^^^^^^^

- Test multiple scenarios
- Track key metrics
- Compare to benchmarks

3. Risk Assessment
^^^^^^^^^^^^^^^

- Monitor drawdowns
- Check recovery ability
- Validate robustness

4. Performance Analysis
^^^^^^^^^^^^^^^^^^^^

- Compare scenario results
- Identify weaknesses
- Improve strategy 