## SpecStory

## 파이썬 Pandas 경고 해결 방법 (2/3/2025, 10:47:02 PM)

_**User**_

테스트가 전부 통과했지만 경고(Warning)들을 정리하고 싶으신 상황이네요.
주요 Warning들을 하나씩 짚어보고 간단히 해결할 수 있는 방법(코드 예시 포함)을 안내드리겠습니다.

1. fillna(method='ffill') / fillna(method='bfill') 관련 FutureWarning
Pandas에서 fillna(method='...') 문법이 더는 권장되지 않으며,
직접 .ffill() 또는 .bfill() 메서드를 호출하라고 안내하고 있습니다.

예:

python
Copy
# (기존)
df['some_col'] = df['some_col'].fillna(method='ffill').fillna(method='bfill')

# (권장)
df['some_col'] = df['some_col'].ffill().bfill()
예시: feature_generator.py
diff
Copy
- features_df[f'SMA_{window}'] = df['$close'].rolling(window=window).mean().fillna(method='ffill').fillna(method='bfill')
+ features_df[f'SMA_{window}'] = df['$close'].rolling(window=window).mean().ffill().bfill()

- features_df[f'EMA_{window}'] = df['$close'].ewm(span=window, adjust=False).mean().fillna(method='ffill').fillna(method='bfill')
+ features_df[f'EMA_{window}'] = df['$close'].ewm(span=window, adjust=False).mean().ffill().bfill()

- features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
+ features_df = features_df.ffill().bfill().fillna(0)
이런 식으로 모두 바꿔주시면 FutureWarning이 사라집니다.

2. 'H' is deprecated and will be removed... 경고
Pandas에서 시간(freq) 단위 'H'(Hours) 대신 'h'를 쓰라는 경고입니다.
가령 df.resample('H') → df.resample('h') 형태로 변경해주시면 됩니다.

diff
Copy
# 예시
- df.resample('H').mean()
+ df.resample('h').mean()
테스트 로그에 'H' is deprecated and will be removed... please use 'h' instead. 라고 나오므로,
시간 관련 코드를 검색해서 'H'를 'h'로 전부 바꿔주시면 됩니다.

일단 이거 두 개만 좀 부탁해!

---

_**Assistant**_

네, 말씀하신 두 가지 Warning을 해결하기 위해 코드를 수정하겠습니다. 먼저 관련 파일들을 찾아보겠습니다.

먼저 `fillna` 관련 경고가 발생하는 파일들을 확인해보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



이제 'H' 시간 포맷 사용에 대한 경고가 발생하는 파일들을 찾아보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



다른 방식으로 시간 관련 코드를 찾아보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



이제 파일들을 읽어보면서 수정이 필요한 부분들을 찾아보겠습니다.

---

_**Assistant**_



---

_**Assistant**_



---

_**Assistant**_



이제 필요한 수정사항들을 적용하겠습니다.

1. `feature_generator.py`의 `fillna` 메서드 사용을 수정하겠습니다:

---

_**Assistant**_

```
// ... existing code ...
        # Check for NaN values
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            self.logger.warning(f"NaN values found in columns: {nan_columns}")
            df = df.ffill().bfill()

// ... existing code ...
            if isinstance(result, pd.Series) and result.isnull().any():
                feature_name = result.name if hasattr(result, 'name') else 'Unknown'
                self.logger.warning(f"NaN values found in {feature_name}, applying forward/backward fill")
                result = result.ffill().bfill().fillna(0)

// ... existing code ...
            # Calculate Moving Averages
            for window in [5, 10, 20]:
                # Simple Moving Average
                features_df[f'SMA_{window}'] = df['$close'].rolling(window=window).mean().ffill().bfill()
                # Exponential Moving Average
                features_df[f'EMA_{window}'] = df['$close'].ewm(span=window, adjust=False).mean().ffill().bfill()

// ... existing code ...
            # Calculate Volume MA
            features_df['volume_ma'] = df['$volume'].rolling(window=20).mean().ffill().bfill()

// ... existing code ...
                features_df[f'CCI_{period}'] = features_df[f'CCI_{period}'].ffill().bfill().fillna(0)

// ... existing code ...
            # Final validation to ensure no NaN values
            nan_columns = features_df.columns[features_df.isna().any()].tolist()
            if nan_columns:
                self.logger.warning(f"NaN values found in columns: {nan_columns}")
                features_df = features_df.ffill().bfill().fillna(0)
```

---

_**Assistant**_



이제 'H' 시간 포맷 사용을 찾아보겠습니다.

---

_**Assistant**_



---

_**Assistant**_







`conftest.py`에서 'H' 시간 포맷을 'h'로 수정하겠습니다:

---

_**Assistant**_

```
// ... existing code ...
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
// ... existing code ...
```

---

_**Assistant**_



이제 주요 경고들이 해결되었습니다. 변경사항을 요약하면:

1. `fillna(method='ffill')`/`fillna(method='bfill')` 관련 FutureWarning 해결:
   - `feature_generator.py`에서 모든 `fillna(method='ffill')`을 `ffill()`로 변경
   - 모든 `fillna(method='bfill')`을 `bfill()`로 변경
   - 연속된 fillna 호출을 체이닝으로 변경 (예: `.ffill().bfill().fillna(0)`)

2. 'H' 시간 포맷 관련 경고 해결:
   - `conftest.py`에서 `freq="H"`를 `freq="h"`로 변경

이러한 변경으로 해당 경고들이 더 이상 발생하지 않을 것입니다. 테스트를 다시 실행하여 변경사항이 잘 적용되었는지 확인하시겠습니까?

---

_**User**_

3. RuntimeWarning: divide by zero encountered in scalar divide
이 메시지는 제로(0)로 나누는 상황에서 발생합니다.
로그에 보면 mean_reversion_ppo_agent.py:195-196 부분:

python
Copy
bb_upper_dist = (bb_upper - current_price) / current_price
bb_lower_dist = (current_price - bb_lower) / current_price
여기서 current_price가 0인 경우가 있는 듯합니다.
안전하게 나누기 연산을 하기 위해서는 0인 경우를 처리해야 합니다.

예시 수정
python
Copy
if current_price == 0:
    bb_upper_dist = 0.0  # 혹은 np.inf로 처리
    bb_lower_dist = 0.0
else:
    bb_upper_dist = (bb_upper - current_price) / current_price
    bb_lower_dist = (current_price - bb_lower) / current_price
또는, 약간의 Epsilon을 두고 나누는 방식도 있습니다:

python
Copy
EPS = 1e-8
bb_upper_dist = (bb_upper - current_price) / max(current_price, EPS)
bb_lower_dist = (current_price - bb_lower) / max(current_price, EPS)


---

_**Assistant**_



---



---

