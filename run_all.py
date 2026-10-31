import json
from pathlib import Path

import pandas as pd

from sim.agent import NimaAgent
from sim.metrics import summarise
from sim.models import load_events, load_scenario
from sim.policies import make_solver_policy, no_redistribution
from sim.simulator import simulate

# يشغّل كل السياسات مع الأحداث نفسها، ويحفظ النتائج ليقرأها التطبيق.
# الوكيل يقرر كل ليلة، لذا يستغرق التشغيل عدة دقائق
# runs every policy with the same events, and saves the results for the app.
# The agent decides every night, so this takes several minutes
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
        # قرار نعمة كل ليلة وسببه
        # Ni'ma's decision each night, and why
        pd.DataFrame(agent.decisions, columns=["day", "priority", "reason", "notes"]).to_csv(
            out / "agent_decisions.csv", index=False)
    m = summarise(sc, log, actions)
    summary[name] = m
    print(f"{name:9s} wasted={m['wasted']:5d}  stockout={m['stockout']:5d}  "
          f"net revenue={m['net_revenue']:,.0f} SAR")

(out / "summary.json").write_text(json.dumps(summary, indent=2))
counts = pd.Series([d["priority"] for d in agent.decisions]).value_counts().to_dict()
print("Ni'ma's nightly choices:", counts)
print("done")