from sim.models import load_scenario
from sim.simulator import simulate
from sim.policies import no_redistribution, make_solver_policy

sc = load_scenario()
policies = {
    "baseline":  no_redistribution,
    "transfers": make_solver_policy("balanced", discount=False, donation=False),
    "value":     make_solver_policy("value"),
    "balanced":  make_solver_policy("balanced"),
    "waste":     make_solver_policy("waste"),
}

for name, policy in policies.items():
    log, actions, _ = simulate(sc, policy, days=30, seed=0)
    t = log.sum(numeric_only=True)
    print(f"{name:9s} wasted={int(t['wasted']):5d}  stockout={int(t['stockout']):5d}  "
          f"sold={int(t['sold']):5d}  clearance={int(t['sold_discount']):4d}  "
          f"donated={int(t['donated']):4d}  transport={actions['cost'].sum():7.1f} SAR")