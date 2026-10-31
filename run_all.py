import json
from pathlib import Path

import pandas as pd

from sim.agent import NimaAgent
from sim.metrics import summarise
from sim.models import load_events, load_scenario
from sim.policies import make_solver_policy, no_redistribution
from sim.simulator import simulate

# يشغّل كل السياسات مع الأحداث نفسها، ويحفظ النتائج ليقرأها التطبيق
# runs every policy with the same events, and saves the results for the app
sc = load_scenario()
events = load_events()
out = Path("runs")
out.mkdir(exist_ok=True)

agent = NimaAgent("balanced")
policies = {
    "baseline": no_redistribution,
    "value":    make_solver_policy("value"),
    "balanced": make_solver_policy("balanced"),
    "waste":    make_solver_policy("waste"),
    "agent":    agent,
}

summary = {}
for name, policy in policies.items():
    log, actions, briefings = simulate(sc, policy, days=30, seed=0, events=events)
    log.to_csv(out / f"{name}_log.csv", index=False)
    actions.to_csv(out / f"{name}_actions.csv", index=False)
    if name == "agent":
        briefings.to_csv(out / "agent_briefings.csv", index=False)
        pd.DataFrame(agent.transcript, columns=["day", "role", "content"]).to_csv(
            out / "agent_transcript.csv", index=False)
    m = summarise(sc, log, actions)
    summary[name] = m
    print(f"{name:9s} wasted={m['wasted']:5d}  stockout={m['stockout']:5d}  "
          f"net revenue={m['net_revenue']:,.0f} SAR")

(out / "summary.json").write_text(json.dumps(summary, indent=2))
print("done")