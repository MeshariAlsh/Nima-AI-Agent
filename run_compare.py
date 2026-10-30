from sim.models import load_scenario
from sim.simulator import simulate
from sim.policies import no_redistribution, solver_only

sc = load_scenario()
cols = ["demand", "sold", "stockout", "wasted"]

for name, policy in [("baseline", no_redistribution), ("solver", solver_only)]:
    log, moves = simulate(sc, policy, days=30, seed=0)
    t = log[cols].sum()
    print(f"{name:9s} wasted={t['wasted']:5d}  stockout={t['stockout']:5d}  "
          f"sold={t['sold']:5d}  moved={int(moves['qty'].sum()):5d}  "
          f"transport={moves['cost'].sum():7.1f} SAR")