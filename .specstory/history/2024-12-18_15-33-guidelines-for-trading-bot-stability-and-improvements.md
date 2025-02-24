## SpecStory

## Guidelines for Trading Bot Stability and Improvements (12/18/2024, 3:33:41 PM)

_**User**_

아래는 새로운 챗에서 cursor에게 줄 수 있는, 매우 자세하고 엄격한 프롬프트 예시입니다. 이 프롬프트는 현재 상황과 이미 해결된 작업, 앞으로 꼭 지켜야 할 규칙, 다음 목표 등을 명확히 제시하며, cursor가 정해진 가이드라인에 따라 작업을 계속하도록 안내합니다. 또한, 이미 안정적으로 작동하는 부분에 대한 수정 금지를 강조하고, 다음에 처리할 우선순위 문제들을 구체적으로 지시합니다.

---

**새로운 챗용 프롬프트 예시**:

**Context & Current State (2024-12-18)**:  
We have made significant progress refactoring the trading bot project according to the development guidelines. Many complex issues have been resolved:

- **MLflow Integration**:  
  Experiment and run lifecycle management stabilized.  
  meta.yaml issues resolved.  
  Logging metrics, parameters, and artifacts now consistent with the tests.

- **Hyperopt / Ray Tune Integration**:  
  Updated from `tune.report()` to `session.report()`.  
  Adjusted `MinimalTuner` and objective functions to produce valid metrics, avoiding `-inf` values.  
  Ensured `evaluate_config()` signature matches test expectations or updated tests accordingly.

- **Risk Backtesting & Data Columns**:  
  Ensured `$`-prefixed OHLCV columns (`$open`, `$high`, `$low`, `$close`, `$volume`) are consistently provided by data generators.  
  Backtester and scenario tests now pass with correct data formatting.

- **Network & Policy Initialization**:  
  PolicyNetwork and ValueNetwork now properly handle (window_size, features) input shapes.  
  Addressed shared attribute issues by either adding the expected layers or updating tests to remove outdated expectations.  
  Ensured consistent activation functions (e.g., Tanh) and proper weight initialization.

- **Multi-Agent Environment & PaperTrading**:  
  MultiAgentTradingEnv reset logic fixed.  
  Observations, actions, and normalization are correctly handled.  
  Trading fee parameter (`trading_fee`) and `$` column usage consistent.  
  Training stability tests now pass.

**All tests are now passing**, indicating that we have a stable baseline that aligns with the development guidelines and test expectations.

---

**Important Rules**:
1. **Do Not Reintroduce Old Issues**:  
   The MLflow, Hyperopt, and Risk Backtesting solutions are stable. **Do not revert or modify these stable parts** without a pressing reason.
   
2. **Adhere Strictly to Development Guidelines**:  
   - `$` prefix for OHLCV columns always.  
   - Use `trading_fee` consistently.  
   - Proper observation and action space handling for all environments.  
   - If tests appear outdated or contradict guidelines, update tests rather than the stable core logic.

3. **Maintain Current Stability**:  
   All tests are passing. Any further optimization or feature additions must not break existing tests. Always run `pytest` after changes.

4. **Document Decisions**:  
   If any future feature or code improvement is needed, document it in comments. Do not break existing stable functionality.

---

**Next Steps**:
Since all tests are currently passing, we are in a stable, MVP-ready state. The next priorities might be:
- Conducting performance optimizations without altering core logic.
- Adding additional logging or monitoring if needed (without breaking tests).
- Implementing optional new features requested by stakeholders (only if aligned with guidelines and without introducing regressions).

For now, no immediate changes are required since all tests pass. If you need to proceed with optional enhancements:
- Start with minor performance profiling (e.g., logging execution times) and ensure no test regressions.
- Consider adding docstrings or comments to clarify code sections that were previously problematic.

If at any point tests fail or you need to adjust logic, return here with the new test outputs and we will refine instructions further.

**Final Instruction to Cursor**:
Since all tests now pass and the system is stable, pause any major refactoring. You may proceed with minor improvements (documentation, micro-optimizations) as long as you do not break existing tests. If new requirements come in, implement them carefully and re-run `pytest` to confirm no regressions occur. If issues arise, provide the new test output and await further instructions.

---

_**Assistant**_

I understand that you've shared the DEVELOPMENT_GUIDELINES.md and README.md files, and mentioned tests/test_risk_backtest.py as relevant. However, I don't see the actual content of test_risk_backtest.py. 

Based on the development guidelines and README, I can see that risk backtesting is a critical component with specific requirements around:

- Column naming (using `$` prefix for OHLCV data)
- Fee parameter naming (using `trading_fee` consistently)
- Data validation rules
- Testing standards

To properly assist with any questions or changes related to test_risk_backtest.py, I would need to see its contents. Could you please share the content of tests/test_risk_backtest.py?

This will help me provide specific guidance while ensuring we maintain:
1. The stable core functionality
2. Consistent naming conventions
3. Proper test coverage
4. Alignment with the development guidelines

---



---

