from sim.models import load_scenario
from sim.simulator import simulate
from sim.policies import no_redistribution, make_solver_policy
from sim.metrics import summarise

sc = load_scenario()
policies = {
    "baseline":      no_redistribution,
    "transfer only": make_solver_policy("balanced", discount=False, donation=False),
    "discount only": make_solver_policy("value", transfer=False, donation=False),
    "donate only":   make_solver_policy("waste", transfer=False, discount=False),
    "value":         make_solver_policy("value"),
    "balanced":      make_solver_policy("balanced"),
    "waste":         make_solver_policy("waste"),
}

print(f"{'policy':13s} {'wasted':>6s} {'stockout':>8s} {'donated':>7s} "
      f"{'revenue':>9s} {'transport':>9s} {'net revenue':>11s}")
for name, policy in policies.items():
    log, actions, _ = simulate(sc, policy, days=30, seed=0)
    m = summarise(sc, log, actions)
    print(f"{name:13s} {m['wasted']:6d} {m['stockout']:8d} {m['donated']:7d} "
          f"{m['revenue']:9.0f} {m['transport']:9.0f} {m['net_revenue']:11.0f}")