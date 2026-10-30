from sim.models import load_scenario
from sim.simulator import simulate
from sim.policies import no_redistribution

sc = load_scenario()
log, actions, _ = simulate(sc, no_redistribution, days=30, seed=0)

cols = ["demand", "sold", "stockout", "wasted"]
print(log.groupby("store")[cols].sum(), "\n")
print(log.groupby("product")[cols].sum(), "\n")
print(log[cols].sum())