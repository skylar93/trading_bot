.. Trading Bot documentation master file

Welcome to Trading Bot's documentation!
=====================================

A comprehensive trading system that combines reinforcement learning, risk management, and real-time execution.

Architecture Overview
-------------------

The trading bot is built with a modular architecture focusing on:

* Reinforcement Learning based trading strategies
* Advanced risk management
* Real-time execution capabilities
* Comprehensive backtesting

Getting Started
-------------

See :doc:`quickstart` for installation and basic usage.

Recent Changes
------------

See :doc:`changelog` for version history and updates.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Project Info
   
   project_structure
   quickstart
   changelog

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/agents
   api/environments
   api/risk_manager
   api/backtest
   api/live_trading

.. toctree::
   :maxdepth: 2
   :caption: Agents & Training
   
   agents/ppo
   agents/networks
   agents/memory

.. toctree::
   :maxdepth: 2
   :caption: Hyperparameter Optimization
   
   hyperopt/ray_tune
   hyperopt/experiments

.. toctree::
   :maxdepth: 2
   :caption: Monitoring & Utilities
   
   utils/logging
   utils/metrics
   utils/visualization

.. toctree::
   :maxdepth: 2
   :caption: Architecture
   
   architecture/overview
   architecture/backtesting/index
   architecture/risk/index
   architecture/live_trading/index

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/training
   examples/backtesting
   examples/live_trading

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

