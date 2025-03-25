# Dimension Mismatch Tasks

This document outlines the specific tasks needed to address the dimension mismatch issues identified in the trading bot codebase.

## Priority 1: Meta Agent Input Dimension Mismatch

### Task 1.1: Investigate Meta Agent Input Processing
- [ ] Locate the meta agent implementation in `agents/strategies/meta_agent.py`
- [ ] Identify where the input reshaping is occurring
- [ ] Determine the expected input shape (10) and the actual input shapes (1650/2190)
- [ ] Analyze why there's such a large discrepancy in dimensions

### Task 1.2: Fix Meta Agent Input Handling
- [ ] Update the meta agent's network architecture to handle the larger input dimensions
- [ ] Alternatively, modify the input preprocessing to reduce dimensions appropriately
- [ ] Add validation to ensure input dimensions are consistent
- [ ] Add unit tests specifically for the meta agent's input processing

## Priority 2: Action Shape Mismatches

### Task 2.1: Standardize Action Spaces
- [ ] Review action space definitions in all environments
- [ ] Ensure consistent action space definitions (e.g., Box(-1, 1, (3,)))
- [ ] Update agent implementations to produce actions with the correct shape
- [ ] Add validation checks for action shapes

### Task 2.2: Fix MultiAgentMultiAssetEnv Action Handling
- [ ] Locate the action processing code in `environments/multi_agent_multi_asset_env.py`
- [ ] Update the action adaptation logic to be more robust
- [ ] Add clear error messages when action shapes don't match expectations
- [ ] Add unit tests for action processing

## Priority 3: Multi-Agent Manager Training Step Errors

### Task 3.1: Debug Input Processing
- [ ] Locate the train_step method in the multi-agent manager
- [ ] Identify why inputs with shape [1, 543] are being passed to agents
- [ ] Determine the correct input shape (batch_size, 13)
- [ ] Add logging to track input shapes throughout the training process

### Task 3.2: Fix Input Reshaping
- [ ] Update the input processing to reshape inputs correctly
- [ ] Add validation checks for input shapes
- [ ] Ensure consistent input shapes across all agents
- [ ] Add unit tests for the train_step method

## Priority 4: Advantages Shape Mismatch

### Task 4.1: Fix Policy Gradient Calculation
- [ ] Locate the policy gradient calculation in the meta agent
- [ ] Update the calculation to handle the dimension mismatch
- [ ] Ensure consistent tensor shapes throughout the training process
- [ ] Add validation checks for tensor shapes

## Testing and Validation

### Task 5.1: Update Shape Verification Tests
- [ ] Enhance shape verification tests to catch dimension mismatches
- [ ] Add specific tests for the meta agent's input processing
- [ ] Add tests for action shape handling
- [ ] Add tests for the multi-agent manager's train_step method

### Task 5.2: Run Validation Script
- [ ] Run the validation script after each fix
- [ ] Verify that the fixes resolve the dimension mismatch issues
- [ ] Document any remaining issues

## Documentation

### Task 6.1: Update Documentation
- [ ] Document the expected tensor shapes for all components
- [ ] Update the architecture documentation to reflect the changes
- [ ] Add examples of correct usage
- [ ] Document the dimension validation process

## Timeline

1. **Week 1**: Address meta agent input dimension mismatch (Priority 1)
2. **Week 2**: Fix action shape mismatches (Priority 2)
3. **Week 3**: Resolve multi-agent manager training step errors (Priority 3)
4. **Week 4**: Fix advantages shape mismatch and complete testing (Priority 4 & 5)
5. **Week 5**: Update documentation and finalize changes (Priority 6)

## Responsible Team Members

- **Meta Agent Issues**: [Assign Team Member]
- **Action Shape Issues**: [Assign Team Member]
- **Multi-Agent Manager Issues**: [Assign Team Member]
- **Testing and Validation**: [Assign Team Member]
- **Documentation**: [Assign Team Member]

## Success Criteria

- All validation tests pass successfully
- No dimension mismatch warnings or errors in the logs
- Consistent tensor shapes throughout the codebase
- Clear documentation of expected tensor shapes
- Robust error handling for dimension mismatches 