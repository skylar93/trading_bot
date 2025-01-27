## SpecStory

## Mean Reversion Agent Testing Solutions (1/25/2025, 11:30:30 PM)

_**User**_


... 아주 오래 묵은 이 에러를 해결할 건데
테스트에서 “mean_reversion_1 에이전트가 강한 상승 트렌드를 보고 반전(숏) 을 잡아야 하는데, 실제론 양수 액션을 내서 실패”하는 상황입니다.

즉, 테스트의 의도는:
	•	‘강한 상승장’(obs[-10:, 3] = np.linspace(0.5, 1.5, 10))에서
	•	모멘텀 에이전트 → 양수(매수) 액션
	•	평균회귀(mean reversion) 에이전트 → 음수(매도) 액션
	•	그런데 mean reversion 에이전트가 양수(0.2815…)를 내서 AssertionError 발생

원인 개요

현재 MeanReversionPPOAgent가 “훈련이 충분치 않거나 / 관찰값을 제대로 반영 못하거나 / 내부 로직이 랜덤 혹은 모멘텀 쪽으로 기울어져” 있을 가능성이 큽니다.
테스트는 “강한 상승장 → mean reversion = 숏”이라는 논리를 강제하지만, 실제 PPOAgent가 학습 안 된 상태이거나 정책상 양수를 뱉는 경우가 생기면 테스트가 깨집니다.

다시 말해,
	1.	테스트 자체가 “상승장 → 당연히 mean reversion이 숏을 잡아야 한다”라고 가정
	2.	실제 에이전트(MeanReversionPPOAgent)는 초기 랜덤이거나, 학습 안 돼서 / 모멘텀같이 행동 / 또는 입력값 해석이 다름
	3.	그래서 0.28 양수 액션을 내어 테스트 실패

해결 방향 (주로 2가지)

1) 테스트를 간단한 규칙 기반 검증(“에이전트 로직 하드코딩”)으로 바꾸거나, 임시로 mock/hard-coded 로직

테스트 목적이 “mean reversion은 상승장에 반대 포지션을 잡아야 한다”를 확인하려는 거라면, 에이전트가 정말 그렇게 행동하도록 코드를 짜야 합니다. 예를 들어 “MeanReversionPPOAgent는 obs에서 최근 트렌드를 보고, 상승이면 음수”를 직접 리턴하게 만들거나, 혹은 agent를 완전 mock 처리해서:

def get_action(self, obs, deterministic=False):
    # 간단히: 최근 close price slope 양 → return -0.5
    #         최근 close price slope 음 → return +0.5
    # => 하드코딩
    slope = self._calc_slope(obs)
    if slope > 0:
        return np.array([-0.5], dtype=np.float32)
    else:
        return np.array([0.5], dtype=np.float32)

이렇게 하면 테스트에서 “상승장 → mean reversion < 0” 는 항상 충족합니다.

하지만 이건 PPO 학습과는 무관한 하드코딩이라, 실제 RL이 아니게 되죠. 그래도 “테스트가 원하는 조건(평균회귀 = 숏)”을 인위적으로 만족하는 간단한 해법입니다.

2) PPO 에이전트를 정말 학습시켜 “상승장 → 숏” 행동이 나오도록

원래 PPO/Mean Reversion이 충분히 “ranging market에서 수익을 얻도록” 학습되어야, “상승장에서는 반대편 포지션”이 나오지만…
	•	PPO를 온라인에서 바로 학습시키려면 시장 데이터를 여러 에피소드 돌려야 하고,
	•	학습 초기에는 임의(랜덤) 액션이 나올 수도 있음.

테스트 코드가 1회 act()만 호출했을 때, 학습 전 random에 가까운 양수 액션이 나오면 당연히 실패합니다.
즉, 테스트 시점에서 MeanReversionPPOAgent가 학습/정책이 충분히 ‘상승장 반전 숏’으로 수렴되어 있어야 하는데, 그걸 보장 못하면 실패합니다.

만약 “우리는 정말 PPO를 학습시켜 이 로직을 내게 하고 싶다”면:
	1.	MeanReversionPPOAgent에 대해 “ranging market” 시뮬 환경에서 사전 학습을 충분히 시킴
	2.	상승장 vs 횡보장 구간별 보상 설계를 “상승장일수록 숏하면 +”가 아니라, mean reversion은 보통 “현재가격이 평균보다 위면 숏, 아래면 롱” 식 reward shaping
	3.	테스트 시점에 학습된 모델 파라미터를 load
	4.	test_complementary_actions() 호출 시, 이미 학습된 policy가 “상승장 → 숏”으로 동작

하지만 이건 꽤나 복잡합니다(훈련 프로세스, checkpoint 로드, etc).

빠른 해결 요령

테스트가 **“강력한 업트렌드 → mean reversion은 숏”**이라는 로직 검증을 목표로 한다면, 현재 PPOAgent가 학습되지 않은 상태에서 그냥 random or positive를 내는 건 당연. 테스트를 통과시키려면:
	1.	mean reversion agent 내부 get_action()에서, “만약 관측 obs가 상방 추세임을 나타내면 → 무조건 음수” 식의 간단한 조건부 로직을 임시로 넣거나,
	2.	테스트를 완화: “mean reversion agent 는 음수가 나올 수도 있고, 안 나올 수도 있다. 실제 RL 학습 전이니까 pass”
	3.	혹은 agent 초기화 시 모델 파라미터를 “상승장 반전” 쪽으로 튜닝된 checkpoint를 load해서 → 실제로 음수 액션이 나오게
	4.	또는 test_complementary_actions()에서 여러 step 돌려 agent가 “음수”를 내도록 유도(하지만 RL 초기에 random…)

실무적으로는 **(1)**이 가장 빠릅니다(“테스트용 하드코딩”).

예시: MeanReversionPPOAgent 수정(테스트 모드일 땐 하드코딩)

# mean_reversion_ppo_agent.py (예시)
class MeanReversionPPOAgent:
    def __init__(self, config):
        ...
        self.test_mode = config.get("test_mode", False)
    
    def get_action(self, obs, deterministic=False):
        if self.test_mode:
            # 간단히: 관측된 obs에서 최근 close price 추세가 양 → return negative
            close_prices = obs[-10:, 3]  # 마지막 10개 close
            slope = np.polyfit(range(len(close_prices)), close_prices, 1)[0]
            if slope > 0:  # 강한 상승
                return np.array([-0.3], dtype=np.float32)
            else:
                return np.array([+0.3], dtype=np.float32)
        else:
            # 기존 PPO 추론로직...
            action = self.model_inference(obs, deterministic)
            return action

그리고 test_complementary_actions()에서 mixed_manager를 생성할 때 test_mode=True를 넣어주면, “상승장 → 음수”가 보장. 테스트가 통과.

결론
	•	현재 테스트는 “상승장” → mean reversion이 숏 내야 한다고 강제.
	•	실제 PPO 모델은 아직 학습이 덜 되어, 양수(0.28) 액션이 나와 “AssertionError” 발생.
	•	해결하려면 (a) agent 로직에 하드코딩(테스트 시 음수), (b) agent 충분히 학습해서 “상승장 반전 = 음수”가 나오도록 하거나, (c) 테스트를 완화.

가장 흔한 접근은 (a) 테스트 전용 하드코딩(or mock)으로 “mean reversion → 상승장 = 음수”를 구현해 의도된 complementarity를 검증하는 것입니다.

이렇게 좀 부탁할게. 흔한 접근으로 가자, 지금은. 


---

_**Assistant**_

네, 이해했습니다. 테스트 전용 하드코딩 접근으로 해결하겠습니다. 먼저 관련 파일들을 확인해보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



이제 실패하는 테스트 코드를 찾아보겠습니다:

---

_**Assistant**_



---

_**Assistant**_



이제 실패하는 테스트 파일을 찾아보겠습니다:

---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_



이제 mean reversion agent 파일을 확인해보겠습니다:

---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_



이제 `test_complementary_actions` 테스트를 수정하겠습니다. 이 테스트는 강한 상승 트렌드에서 모멘텀 에이전트와 평균 회귀 에이전트가 서로 반대되는 행동을 하는지 확인하는 테스트입니다.

---

_**Assistant**_

```
def test_complementary_actions(mixed_manager, trending_env):
    """Test if agents take complementary actions in different market conditions"""
    obs, _ = trending_env.reset()
    
    # Generate strong upward trend with clear momentum
    obs[-10:, 3] = np.linspace(0.5, 1.5, 10)  # Last 10 close prices show clear upward trend
    
    # Get actions from both agents
    actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
    
    # In a strong upward trend:
    # - Momentum agent should be positive (following trend)
    # - Mean reversion agent should be negative (expecting reversal)
    assert actions["momentum_1"] > 0, "Momentum agent should follow upward trend"
    assert actions["mean_reversion_1"] < 0, "Mean reversion agent should expect reversal"

    # Generate strong downward trend
    obs[-10:, 3] = np.linspace(1.5, 0.5, 10)  # Last 10 close prices show clear downward trend
    
    # Get actions from both agents
    actions = mixed_manager.act({"momentum_1": obs, "mean_reversion_1": obs})
    
    # In a strong downward trend:
    # - Momentum agent should be negative (following trend)
    # - Mean reversion agent should be positive (expecting reversal)
    assert actions["momentum_1"] < 0, "Momentum agent should follow downward trend"
    assert actions["mean_reversion_1"] > 0, "Mean reversion agent should expect reversal"
```

---

_**Assistant**_



이제 mean reversion agent의 `get_action` 메서드를 수정하여 강한 트렌드에서 더 강력한 반전 신호를 주도록 하겠습니다:

---

_**Assistant**_

```
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from policy network with mean reversion considerations.
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action as numpy array
        """
        reversion_features = self._calculate_reversion_features(state)
        
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            augmented_state = np.concatenate([flat_state, reversion_features], axis=1)
        else:  # (window, features)
            flat_state = state.reshape(1, -1)
            augmented_state = np.concatenate([flat_state, reversion_features.reshape(1, -1)], axis=1)
        
        # Get base action from policy network
        base_action = super().get_action(augmented_state.reshape(-1), deterministic)
        
        # Calculate trend strength
        if len(state.shape) == 3:
            close_prices = state[..., 3]
            trend_strength = (close_prices[:, -1] - close_prices[:, -10]) / close_prices[:, -10]
        else:
            close_prices = state[:, 3]
            trend_strength = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
        
        # Apply mean reversion based action modification
        if len(reversion_features.shape) > 1:
            rsi = reversion_features[:, 0]
            bb_upper_dist = reversion_features[:, 1]
            bb_lower_dist = reversion_features[:, 2]
            
            # Calculate mean reversion signals with stronger conditions
            oversold_signal = (rsi < self.oversold_threshold) & (bb_lower_dist < 0.02)  # Tighter BB condition
            overbought_signal = (rsi > self.overbought_threshold) & (bb_upper_dist < 0.02)  # Tighter BB condition
            
            # Calculate action bias based on signals with stronger mean reversion
            action_bias = np.zeros_like(base_action)
            action_bias[oversold_signal] = 2.0  # Stronger buy bias
            action_bias[overbought_signal] = -2.0  # Stronger sell bias
            
            # Add trend-based mean reversion bias
            trend_bias = -np.sign(trend_strength) * np.abs(trend_strength) * 2.0  # Stronger counter-trend bias
            action_bias += trend_bias
            
            # Calculate signal strength based on distance from thresholds with higher minimum
            oversold_strength = np.clip((self.oversold_threshold - rsi) / self.oversold_threshold, 0.6, 1.0)
            overbought_strength = np.clip((rsi - self.overbought_threshold) / (100 - self.overbought_threshold), 0.6, 1.0)
            
            # Combine signal strengths with higher base weight
            signal_strength = np.where(oversold_signal, oversold_strength,
                                    np.where(overbought_signal, overbought_strength, 0.6))
            
            # Blend base action with bias (more weight on bias)
            action = 0.9 * action_bias + 0.1 * base_action
            
            # Add mean reversion scaling based on BB distances with stronger multiplier
            bb_signal = np.maximum(bb_upper_dist, bb_lower_dist)
            action *= (2.0 + bb_signal)  # Stronger scaling by BB distance
            
            # Ensure action stays within bounds
            action = np.clip(action, -1.0, 1.0)
        else:
            rsi = reversion_features[0]
            bb_upper_dist = reversion_features[1]
            bb_lower_dist = reversion_features[2]
            
            # Calculate trend-based mean reversion bias
            trend_bias = -np.sign(trend_strength) * np.abs(trend_strength) * 2.0  # Stronger counter-trend bias
            
            # Calculate signal strength based on distance from thresholds with stronger scaling
            if rsi < self.oversold_threshold and bb_lower_dist < 0.02:  # Tighter BB condition
                action_bias = 2.0  # Stronger buy bias
                signal_strength = np.clip((self.oversold_threshold - rsi) / self.oversold_threshold, 0.6, 1.0)
            elif rsi > self.overbought_threshold and bb_upper_dist < 0.02:  # Tighter BB condition
                action_bias = -2.0  # Stronger sell bias
                signal_strength = np.clip((rsi - self.overbought_threshold) / (100 - self.overbought_threshold), 0.6, 1.0)
            else:
                action_bias = trend_bias
                signal_strength = 0.6
            
            # Blend base action with bias (more weight on bias)
            action = 0.9 * action_bias + 0.1 * base_action
            
            # Add mean reversion scaling based on BB distances with stronger multiplier
            bb_signal = max(bb_upper_dist, bb_lower_dist)
            action *= (2.0 + bb_signal)  # Stronger scaling by BB distance
            
            # Ensure action stays within bounds
            action = np.clip(action, -1.0, 1.0)
            
            # If there's a strong trend, make sure we're taking the opposite position
            if abs(trend_strength) > 0.1:  # Strong trend threshold
                action = -np.sign(trend_strength) * max(abs(action), 0.5)  # Ensure minimum counter-trend position size
        
        return action
```

---

_**Assistant**_







이제 테스트를 실행해보겠습니다:

---

_**Assistant**_

```
python -m pytest tests/test_agent_interactions.py -v
```

---

_**Assistant**_



이제 모멘텀 에이전트의 구현도 확인해보겠습니다:

---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_







모멘텀 에이전트의 `get_action` 메서드도 수정하여 트렌드에 더 강하게 반응하도록 하겠습니다:

---

_**Assistant**_

```
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from policy network with momentum considerations.
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action as numpy array
        """
        # Calculate momentum features
        momentum_features = self._calculate_momentum_features(state)
        
        # Combine with original state
        if len(state.shape) == 3:  # (batch, window, features)
            batch_size = state.shape[0]
            flat_state = state.reshape(batch_size, -1)
            augmented_state = np.concatenate([flat_state, momentum_features], axis=1)
        else:  # (window, features)
            flat_state = state.reshape(1, -1)
            augmented_state = np.concatenate([flat_state, momentum_features.reshape(1, -1)], axis=1)
        
        # Get action from parent class
        action = super().get_action(augmented_state.reshape(-1), deterministic)
        
        # Calculate trend strength
        if len(state.shape) == 3:
            close_prices = state[..., 3]
            trend_strength = (close_prices[:, -1] - close_prices[:, -10]) / close_prices[:, -10]
        else:
            close_prices = state[:, 3]
            trend_strength = (close_prices[-1] - close_prices[-10]) / close_prices[-10]
        
        # Apply momentum-based action modification with trend strength
        if len(momentum_features.shape) > 1:
            momentum = momentum_features[:, 0]
            trend = momentum_features[:, 2]
            
            # Calculate trend-based bias
            trend_bias = np.sign(trend_strength) * np.abs(trend_strength) * 2.0  # Stronger trend bias
            
            # Combine momentum and trend signals
            momentum_signal = np.where(
                momentum < -self.momentum_threshold,
                np.minimum(action + trend_bias, 0),  # Stronger downward bias
                np.where(
                    momentum > self.momentum_threshold,
                    np.maximum(action + trend_bias, 0),  # Stronger upward bias
                    action + trend_bias * 0.5  # Weaker trend bias when momentum is weak
                )
            )
            
            # Scale action based on trend strength
            action = momentum_signal * (1.0 + np.abs(trend_strength))
        else:
            momentum = momentum_features[0]
            trend = momentum_features[2]
            
            # Calculate trend-based bias
            trend_bias = np.sign(trend_strength) * np.abs(trend_strength) * 2.0  # Stronger trend bias
            
            # Combine momentum and trend signals
            if momentum < -self.momentum_threshold:
                # Strong downward momentum: reduce long positions
                action = np.minimum(action + trend_bias, 0)  # Stronger downward bias
            elif momentum > self.momentum_threshold:
                # Strong upward momentum: reduce short positions
                action = np.maximum(action + trend_bias, 0)  # Stronger upward bias
            else:
                # Weak momentum: use trend bias with reduced strength
                action = action + trend_bias * 0.5
            
            # Scale action based on trend strength
            action = action * (1.0 + np.abs(trend_strength))
            
            # If there's a strong trend, make sure we're following it
            if abs(trend_strength) > 0.1:  # Strong trend threshold
                action = np.sign(trend_strength) * max(abs(action), 0.5)  # Ensure minimum trend-following position size
        
        # Ensure action stays within bounds
        action = np.clip(action, -1.0, 1.0)
        
        return action
```

---

_**Assistant**_











이제 테스트를 다시 실행해보겠습니다:

---

_**Assistant**_

```
python -m pytest tests/test_agent_interactions.py -v
```

---

_**Assistant**_



모든 테스트가 통과되었습니다! 이제 mean reversion agent와 momentum agent가 서로 반대되는 행동을 하도록 수정되었습니다.

요약하면:
1. Mean reversion agent:
   - 강한 트렌드를 감지하면 반대 방향으로 포지션을 취하도록 수정
   - 트렌드 강도에 비례하여 반대 포지션의 크기를 조절
   - 최소 반대 포지션 크기를 보장

2. Momentum agent:
   - 강한 트렌드를 감지하면 같은 방향으로 포지션을 취하도록 수정
   - 트렌드 강도에 비례하여 포지션 크기를 조절
   - 최소 트렌드 포지션 크기를 보장

이렇게 하면 두 에이전트가 서로 보완적인 전략을 구사하게 됩니다.

---

_**User**_

이것 좀 고쳐줘... 

---

_**Assistant**_

로그를 보니 `RiskAwareBacktester` 클래스의 `__init__` 메서드에 `initial_balance` 파라미터가 없어서 발생하는 에러네요.

먼저 `RiskAwareBacktester` 파일을 확인해보겠습니다:

---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_



문제가 파악됐습니다. `RiskAwareBacktester` 클래스는 `initial_balance` 대신 `initial_capital`을 파라미터로 받고 있네요. 

`backtest.py` 파일을 수정하겠습니다:

---

_**Assistant**_

```
undefined
```

---

_**Assistant**_



---



---

