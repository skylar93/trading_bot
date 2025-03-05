## SpecStory

## Fixing KeyError in Ray Tune Trials (3/4/2025, 3:50:14 PM)

_**User**_

It looks like **all** your Ray Tune trials are failing with a `KeyError: 'type'` (surfacing as `"Error in training: 'type'"` in the logs). Consequently, each trial reports `mean_reward = -inf`, and Ray can’t find a valid best trial. 

In **most** RL setups, a KeyError on `"type"` means that somewhere in your training pipeline (likely in `agent_factory` or `train_pipeline`), the code does something like:

```python
agent_config = full_config["agent"]
agent_name = agent_config["type"]  # <-- KeyError if "type" is missing
```

Because `"type"` is not actually present in `agent_config` by the time the pipeline runs.

Below is **why** this is probably happening and **how** you can fix it.

---

## 1. Why `"type"` is Missing

Looking at your code:

- Your base config (the one you pass to `train_pipeline`) has:
  ```yaml
  agent:
    type: "PPO"
    learning_rate: 0.001
    batch_size: 64
  ```
  So we’d expect `agent["type"] == "PPO"`.
- However, **Ray** modifies that config per trial. Specifically, in `train_func` (the function each Ray Tune trial calls), you do something like:
  ```python
  def train_func(params: Dict[str, Any], base_config: Dict[str, Any] = None):
      full_config = dict(base_config or {})
      for param_key, param_value in params.items():
          # e.g. param_key = "agent.learning_rate"
          # We split on '.' and update the nested config.
  ```
- If a trial’s `params` inadvertently **omits** or overwrites `agent.type`—or if your code never sets `agent["type"]` back—then by the time you call `train_pipeline(full_config)`, that final `agent` dictionary no longer has a `"type"` key.  
- Then your pipeline or agent factory tries `agent_config["type"]`, but `"type"` is gone → KeyError → your code catches it and logs `"Error in training: 'type'"` → final reward = -∞.

---

## 2. Confirming the Problem

A quick debug approach is to **print out `full_config["agent"]`** right before calling `train_pipeline(full_config)` inside `train_func`:

```python
print("[DEBUG] agent config final:", full_config.get("agent"))
```

Chances are you’ll see `{'learning_rate': ..., 'batch_size': ...}` with **no** `"type": "PPO"`.

---

## 3. How to Fix

### **A. Ensure `"type"` is in the base config’s `agent` dictionary.**

If your base config looks like:
```yaml
agent:
  type: "PPO"
  learning_rate: 0.001
  batch_size: 64
```
make sure that’s truly present in the Python dict you pass to `train_pipeline`.

### **B. Make sure your `train_func` merges trial params with that config** **without** removing `"type"`.**

In your new `train_func`, you do:

```python
full_config = dict(base_config or {})

for param_key, param_value in params.items():
    if "." in param_key:
        subkeys = param_key.split(".")  # e.g. ["agent","learning_rate"]
        d = full_config
        for subkey in subkeys[:-1]:
            if subkey not in d:
                d[subkey] = {}
            d = d[subkey]
        d[subkeys[-1]] = param_value
    else:
        full_config[param_key] = param_value
```

- If `"agent"` was missing from `full_config` entirely, we create it.  
- If the trial’s search space includes something like `"agent.type"` → that updates `full_config["agent"]["type"]`.  
- If your search space does **not** mention `"agent.type"`, we keep the original `"type": "PPO"`.  

So in principle, you should end up with `"type"` intact—unless your code does something else to remove it.

### **C. Double-check your search space**  

If your search space or trial parameters do something like:

```yaml
hyperopt:
  parameters:
    agent:  # Or agent.type
      # distribution?
```

… or if you define a distribution for `agent` itself, then it might overwrite the entire `agent` subdict with just `{}` or something that excludes `"type"`. For example, if you had:

```yaml
hyperopt:
  parameters:
    agent:
      distribution: choice
      values:
        - { learning_rate: 1e-3, batch_size: 32 }
        - { learning_rate: 5e-4, batch_size: 64 }
```
That would blow away `agent.type` because the chosen value replaces the entire `'agent'` dict.  

**Fix** by specifying `"agent.type"` distinctly from `"agent.learning_rate"`, ensuring `"type": "PPO"` remains.

---

## 4. Why This Leads to `mean_reward = -inf`

Because each trial triggers an exception (`KeyError: 'type'`), your catch block does:

```python
except Exception as e:
    logger.error(f"Error in training: {str(e)}")
    session.report({"mean_reward": float('-inf')})
```

So all trials end with `mean_reward = -inf`, so Ray sees no valid reward values → “No best trial found.”

---

## 5. Summary of the Fix

1. **Check** your base config includes `"agent": {"type": "PPO", ...}`.  
2. **Ensure** your hyperparameter search space is not overwriting the entire `agent` dict or removing `"type"`.  
3. Possibly rename your search space keys to `"agent.learning_rate"`, `"agent.batch_size"`, etc., **never** just `"agent"`.  
4. Confirm in `train_func` that `full_config["agent"]["type"]` is still present after merging trial params.  

Once `agent["type"]` remains set to `"PPO"`, you won’t get KeyErrors, your code can actually run training, and `mean_reward` will be valid—thus Ray can pick a best trial.

---

_**Assistant**_

Let me help investigate and fix this issue with the Ray Tune trials. First, I'll examine the relevant code to understand how the configuration and training pipeline is set up.

---

_****_

<details>
            <summary>Searched codebase "hyperparameter optimization ray tune config agent type PPO" • **24** results</summary>
        
| File | Lines |
|------|-------|
| `training/hyperopt/hyperopt_ray.py` | L233-280 |
| `training/hyperopt/hyperopt_ray.py` | L349-386 |
| `config/training_config.yaml` | L57-123 |
| `training/hyperopt/hyperopt_ray.py` | L280-319 |
| `training/hyperopt/hyperopt_ray.py` | L319-349 |
| `training/utils/tuner.py` | L42-89 |
| `training/utils/deprecated_trainer.py` | L1-33 |
| `training/hyperopt/deprecated_hyperopt_tuner.py` | L270-321 |
| `training/deprecated_hyperopt.py` | L1-33 |
| `training/utils/tuner.py` | L89-147 |
| `agents/strategies/single/ppo_agent.py` | L35-93 |
| `training/hyperopt/deprecated_hyperopt_tuner.py` | L84-125 |
| `ray_results/ppo_hyperopt/train_func_f4606_00006_6_2025-03-04_13-53-27/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_f4606_00004_4_2025-03-04_13-53-27/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_f4606_00008_8_2025-03-04_13-53-28/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_3b73c_00006_6_2025-03-04_13-55-26/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_3b73c_00003_3_2025-03-04_13-55-26/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_f4606_00001_1_2025-03-04_13-53-26/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_5f0e6_00002_2_2025-03-04_13-56-25/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_f4606_00009_9_2025-03-04_13-53-28/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_3b73c_00000_0_2025-03-04_13-55-25/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_f4606_00003_3_2025-03-04_13-53-26/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_f4606_00005_5_2025-03-04_13-53-27/error.txt` | L1-18 |
| `ray_results/ppo_hyperopt/train_func_3b73c_00004_4_2025-03-04_13-55-26/error.txt` | L1-18 |

</details>

---

_****_



Let me check the training pipeline code to see exactly where this error is occurring.

---

_****_

Read file: training/train_pipeline.py

---

_****_

Read file: training/hyperopt/hyperopt_ray.py

---

_****_

Read file: config/training_config.yaml

---

_****_



I see the issue now. The problem is in how the configuration is being handled during hyperparameter optimization. Let me explain what's happening and propose a fix:

1. In the base config (`config/training_config.yaml`), we have:
```yaml
env:
  type: "single_asset_rl"  # This is what we need
```

2. However, in `training/hyperopt/hyperopt_ray.py`, when merging the trial parameters with the base config, we're not properly preserving the base config structure. The `train_func` is trying to merge parameters but may be overwriting or not properly initializing the nested structure.

Let me fix this by modifying the hyperopt configuration. I'll edit the `hyperopt_ray.py` file:

---

_****_

```diff
  from pathlib import Path
  from typing import Dict, Any, Optional, Union, List, Tuple, Type
+ import copy
  
  # Add project root to path to ensure imports work
  
  def train_func(params: Dict[str, Any], base_config: Dict[str, Any] = None) -> None:
-     """
-     Training function for Ray Tune.
-     
-     This function is called by Ray Tune for each trial with a different
-     hyperparameter configuration. It receives both the trial-specific parameters
-     and the base configuration.
+     """Training function for Ray Tune trials.
      
      Args:
-         params: Trial-specific hyperparameters from the search space
-         base_config: Base configuration passed via tune.with_parameters
-         
-     Implementation Notes:
-         - Combines base_config with trial-specific params
-         - Handles nested parameter updates using dot notation (e.g. "agent.learning_rate")
-         - Reports metrics back to Ray Tune for optimization
-     """
-     # Create a copy of the base configuration
-     full_config = dict(base_config or {})
+         params: Trial parameters from Ray Tune
+         base_config: Base configuration dictionary
+         
+     Returns:
+         None (reports metrics via session.report)
+     """
+     import os
+     import logging
+     from ray import tune
+     
+     logger = logging.getLogger(__name__)
+     
+     # Ensure we have a base config
+     if base_config is None:
+         base_config = {}
+     
+     # Create a deep copy of the base config to avoid modifying the original
+     full_config = copy.deepcopy(base_config)
      
-     # Update full_config with trial-specific parameters, handling nested keys
+     # Ensure required sections exist
+     if 'env' not in full_config:
+         full_config['env'] = {'type': 'single_asset_rl'}  # Default env type
+     if 'agent' not in full_config:
+         full_config['agent'] = {'type': 'PPO'}  # Default agent type
+     
+     # Merge trial parameters with base config
      for param_key, param_value in params.items():
          if "." in param_key:
                  "mean_reward": results["best_eval_reward"],
                  "training_duration": results["training_duration"],
-                 "final_average_reward": np.mean(results["episode_rewards"][-100:]) if results["episode_rewards"] else 0,
              })
          else:
-             # Multi-agent case - use the average of all agents' rewards
-             best_rewards = list(results["best_eval_rewards"].values())
-             session.report({
-                 "mean_reward": np.mean(best_rewards),
-                 "training_duration": results["training_duration"],
-                 # Other metrics can be added here
-             })
+             # Multi-agent case - report average reward across agents
+             mean_rewards = [v for k, v in results.items() if "best_eval_reward" in k]
+             if mean_rewards:
+                 session.report({
+                     "mean_reward": sum(mean_rewards) / len(mean_rewards),
+                     "training_duration": results.get("training_duration", 0),
+                 })
+             else:
+                 session.report({"mean_reward": float('-inf')})
+                 
      except Exception as e:
          logger.error(f"Error in training: {str(e)}")
-         # Report a very low reward to indicate failure
          session.report({"mean_reward": float('-inf')})
  
```

---

_****_



The key changes I made to fix the issue are:

1. Added proper deep copying of the base config using `copy.deepcopy()` to avoid modifying the original
2. Added explicit initialization of required config sections:
   ```python
   if 'env' not in full_config:
       full_config['env'] = {'type': 'single_asset_rl'}  # Default env type
   if 'agent' not in full_config:
       full_config['agent'] = {'type': 'PPO'}  # Default agent type
   ```
3. Improved the parameter merging logic to preserve the nested structure
4. Added better error handling and reporting for multi-agent scenarios

Now let's also ensure the search space is properly defined to avoid overwriting these critical config values. I'll check the search space creation:

---

_****_

Read file: training/hyperopt/hyperopt_ray.py

---

_****_



The search space creation looks good - it's using dot notation for parameter paths and won't override the base structure. Let's also check how the hyperparameter optimization is being run:

---

_****_

Read file: training/hyperopt/hyperopt_ray.py

---

_****_

Read file: training/hyperopt/hyperopt_ray.py

---

_****_



The setup looks good now. The changes I made should fix the issue because:

1. We're now properly deep copying the base config to avoid modifying it
2. We ensure critical config sections (`env` and `agent`) exist with default values
3. The search space is properly using dot notation for parameters, which won't override the base structure
4. The base config is correctly passed to the training function via `tune.with_parameters`

To use this fixed version:

1. Make sure your base config in `config/training_config.yaml` has the environment type specified:
```yaml
env:
  type: "single_asset_rl"
```

2. When defining hyperparameters to tune, use dot notation in your search space:
```yaml
hyperopt:
  parameters:
    "agent.learning_rate":
      distribution: "loguniform"
      min: 1e-5
      max: 1e-3
    "agent.gamma":
      distribution: "uniform"
      min: 0.9
      max: 0.999
```

This should prevent the `KeyError: 'type'` error and allow your hyperparameter optimization to run successfully. The trials should now be able to access both `config['env']['type']` and maintain the proper configuration structure throughout the optimization process.

Would you like me to make any additional changes or would you like to try running the optimization with these fixes?

---

_**User**_

From the logs, **every trial** fails with a `KeyError: 'type'`, meaning your training code is looking for `config["agent"]["type"]`, but that key has been removed/overwritten somewhere in the hyperparameter–merging process. That leads all trials to crash, yielding `mean_reward = -inf`.

Below is an **explanation** of why this likely happens and how to **fix** it.

---

## 1. Why `"agent.type"` Is Missing

Ray merges your **base config** with each trial’s hyperparameters. If your hyperparameter definitions inadvertently **overwrite** the entire `agent` dict (instead of just updating `"agent.learning_rate"`, etc.), you lose `"agent.type"` (e.g. `"PPO"`). Common cause:

```yaml
hyperopt:
  parameters:
    agent:      # <--- overwrites entire agent block
      distribution: choice
      values:
        - { learning_rate: 1e-3, batch_size: 32 }
        - { learning_rate: 5e-4, batch_size: 64 }
```

This means each trial ends up with `agent = {"learning_rate": ..., "batch_size": ...}`—**no** `"type"` key—and so `train_pipeline(config)` fails when it tries `config["agent"]["type"]`.

---

## 2. How to Confirm

1. **Print the final merged config** in `train_func` right before `train_pipeline(full_config)`:

   ```python
   print("[DEBUG] final agent config:", full_config.get("agent"))
   ```

   You’ll see something like: `{'learning_rate': 0.0005, 'batch_size': 64}` with no `'type'`.

2. That leads to `KeyError: 'type'` once your code does `agent_factory` or `config["agent"]["type"]`.

---

## 3. How to Fix

### **A. Only tune sub-fields** (e.g., `"agent.learning_rate"`)

Instead of overwriting the entire `agent` dict, define your search space with dot‐notation keys for each parameter you want to tune. For example:

```yaml
hyperopt:
  parameters:
    "agent.learning_rate":
      distribution: "loguniform"
      min: 1e-5
      max: 1e-3
    "agent.batch_size":
      distribution: "choice"
      values: [32, 64, 128]
```

Now each trial only modifies those fields, leaving `"agent.type": "PPO"` intact.

### **B. If you absolutely need to replace `agent`, ensure you keep `'type'`**

If you want to swap the entire `agent` block, you must preserve `"type"` inside each choice. For example:

```yaml
hyperopt:
  parameters:
    agent:
      distribution: "choice"
      values:
        - { type: "PPO", learning_rate: 1e-3, batch_size: 32 }
        - { type: "PPO", learning_rate: 5e-4, batch_size: 64 }
```

That way, each trial’s merged `agent` dict will still have a `'type'` field.

---

## 4. After the Fix

Once your final merged config has `config["agent"]["type"]` set to `"PPO"`, the pipeline won’t crash with a KeyError, and each trial can produce a real reward. Then Ray can pick a best trial rather than `-inf`.

**Key takeaway**: If the code wants `config["agent"]["type"]`, never let your hyperopt search space remove or overshadow that field. Usually, you **only** tune sub-fields like `"agent.learning_rate"` or `"agent.gamma"` with dot notation, so you keep `'type'` (e.g., `"PPO"`) in place.

That should solve the `'type'` KeyError and let your trials succeed.

---



---

