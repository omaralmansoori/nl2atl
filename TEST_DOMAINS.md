# Test Domains vs Training Domains

## Purpose

To properly evaluate model generalization, the test set uses **different domains** than those in the training data.

## Training Domains (Used for Fine-tuning)

The following domains were used in the training data:

1. **autonomous_systems** - Drone controllers, autopilots, navigation
2. **distributed_systems** - Leader election, consensus, replication
3. **safety_critical** - Safety monitors, hazard detection, emergency control
4. **robotics** - Robot arms, grippers, motion planning
5. **access_control** - Authentication, authorization, session management
6. **industrial_control** - Cooling systems, temperature regulation, pressure control
7. **networking** - Firewalls, packet routing, load balancing
8. **transaction_processing** - Transaction managers, commit/rollback, lock management

## Test Domains (Used for Evaluation)

The following NEW domains are used for testing to evaluate generalization:

1. **healthcare** - Patient monitoring, drug dispensing, vital signs
2. **smart_home** - Thermostats, security systems, lighting control
3. **autonomous_vehicles** - Cruise control, braking, lane keeping
4. **supply_chain** - Inventory management, shipping, quality control
5. **financial_systems** - Fraud detection, payment processing, risk analysis
6. **energy_grid** - Grid balancing, power distribution, renewable integration
7. **air_traffic_control** - Flight control, radar, runway management
8. **manufacturing** - Assembly lines, quality inspection, robotic welding

## Why This Matters

Testing on different domains evaluates:

- **Generalization**: Can the fine-tuned model handle domains it has never seen?
- **Robustness**: Does the model rely on domain-specific patterns or learn general ATL translation?
- **Overfitting**: Has the model memorized training domains or learned transferable skills?

## Running Experiments

### Default (Test Domains)
```bash
# Uses test domains by default
python full_model_comparison.py --count 100 --verbose
```

### Training Domains (for comparison)
```bash
# Use training domains explicitly
python full_model_comparison.py --count 100 --use-training-domains --verbose
```

### Specific Domains
```bash
# Test on specific subset of test domains
python full_model_comparison.py --count 50 --domains healthcare,smart_home
```

## Configuration Files

- **Training domains**: `config/templates_atl.json` (domains section)
- **Test domains**: `config/test_domains.json` (test_domains section)

Both are automatically merged when loading, so the generator can use either set.

## Expected Results

A well-generalized model should:
- Maintain high accuracy on test domains (≥80%)
- Show similar performance to training domains
- Handle new agent names and propositions correctly
- Apply ATL operators appropriately regardless of domain

Significant performance drop on test domains would indicate overfitting to training domains.
