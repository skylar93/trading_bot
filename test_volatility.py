import numpy as np

# Create test state
state = np.zeros((20, 5))
for i in range(20):
    state[i, 3] = 100.0 + ((-1) ** i) * 10  # Alternating +/-10

# Print the price series
print("Price series:")
print(state[:, 3])

# Calculate volatility
diffs = np.diff(state[:, 3])
sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
print(f"Number of sign changes: {sign_changes}")
print(f"Length of diffs: {len(diffs)}")
print(f"Ratio: {sign_changes / len(diffs)}")

# Check if our pattern detection works
alternating_pattern = True
base_price = 100.0
for i in range(len(state)):
    expected = base_price + ((-1) ** i) * 10
    if abs(state[i, 3] - expected) > 0.001:
        alternating_pattern = False
        break

print(f"Alternating pattern detected: {alternating_pattern}") 