import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from sim.models import load_scenario
from sim.agent import live_plan
from sim.models import load_events
from sim.simulator import ACTION_COLS

st.set_page_config(page_title="Ni'ma", page_icon="🌿", layout="wide")

RUNS = Path("runs")
# أسماء الأولويات كما يراها المدير
# priority names as the manager sees them
PRIORITY_LABELS = {"value": "Recover value (optimiser)",
                   "balanced": "Balanced (optimiser)",
                   "waste": "Reduce waste (optimiser)",
                   "agent": "Ni'ma AI agent (balanced + notes)"}
BLUE = [30, 120, 220, 200]
GREEN = [40, 170, 90, 200]
RED = [220, 60, 60, 200]
GREY = [120, 120, 120, 160]


@st.cache_data
def load_run(name: str):
    """
    يقرأ سجلّ المحاكاة والإجراءات المحفوظة لسياسة واحدة.
    Reads the saved simulation log and actions for one policy.
    """
    log = pd.read_csv(RUNS / f"{name}_log.csv")
    actions = pd.read_csv(RUNS / f"{name}_actions.csv")
    return log, actions


@st.cache_data
def load_summary() -> dict:
    return json.loads((RUNS / "summary.json").read_text())


sc = load_scenario()
# إحداثيات كل متجر وجهة مستفيدة
# coordinates of every store and recipient
LOC = {s.name: [s.lon, s.lat] for s in sc.stores}
LOC.update({r.name: [r.lon, r.lat] for r in sc.recipients})


# ---------- الخريطة ----------
# ---------- the map ----------
def store_points(log: pd.DataFrame, day: int) -> pd.DataFrame:
    """
    دوائر المتاجر: يكبر حجمها ويحمرّ لونها مع الهدر.
    Store circles: they grow and turn red with waste.
    """
    d = (log[log["day"] == day]
         .groupby("store")[["wasted", "stockout", "sold"]].sum().reset_index())
    d["position"] = d["store"].map(LOC)
    d["radius"] = 300 + d["wasted"] * 25
    d["color"] = d["wasted"].apply(lambda w: RED if w > 0 else GREY)
    d["label"] = d.apply(lambda r: f"{r.store}\nwasted: {r.wasted}  "
                                   f"missed sales: {r.stockout}", axis=1)
    return d


def action_arcs(actions: pd.DataFrame, day: int) -> pd.DataFrame:
    """
    أقواس النقل (أزرق) والتبرع (أخضر) لهذا اليوم.
    Transfer (blue) and donation (green) arcs for this day.
    """
    a = actions[(actions["day"] == day)
                & (actions["type"].isin(["transfer", "donation"]))]
    if a.empty:
        return a
    a = a.groupby(["type", "src", "dst"], as_index=False)["qty"].sum()
    a["source"] = a["src"].map(LOC)
    a["target"] = a["dst"].map(LOC)
    a["color"] = a["type"].map({"transfer": BLUE, "donation": GREEN})
    a["width"] = 2 + a["qty"] / 15
    a["label"] = a.apply(lambda r: f"{r.type}: {r.qty} units\n{r.src} → {r.dst}", axis=1)
    return a


def make_map(log: pd.DataFrame, actions: pd.DataFrame, day: int) -> pdk.Deck:
    stores = store_points(log, day)
    recipients = pd.DataFrame([{"position": [r.lon, r.lat], "label": r.name}
                               for r in sc.recipients])
    layers = [
        pdk.Layer("ScatterplotLayer", data=stores,
                  get_position="position", get_radius="radius",
                  get_fill_color="color", pickable=True),
        # أسماء المتاجر على الخريطة
        # store names on the map
        pdk.Layer("TextLayer", data=stores,
                  get_position="position", get_text="store",
                  get_size=14, get_color=[255, 255, 255],
                  get_alignment_baseline="'bottom'", get_pixel_offset=[0, -12]),
        pdk.Layer("ScatterplotLayer", data=recipients,
                  get_position="position", get_radius=350,
                  get_fill_color=GREEN, pickable=True),
    ]
    arcs = action_arcs(actions, day)
    if not arcs.empty:
        layers.append(pdk.Layer("ArcLayer", data=arcs,
                                get_source_position="source",
                                get_target_position="target",
                                get_source_color="color", get_target_color="color",
                                get_width="width", pickable=True))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=24.69, longitude=46.72,
                                         zoom=9.8, pitch=20),
        map_style=None,
        tooltip={"text": "{label}"},
    )


# ---------- الشريط الجانبي ----------
# ---------- sidebar ----------
st.sidebar.title("Ni'ma  نعمة")
st.sidebar.caption("An AI agent that keeps food from going to waste across a grocery chain.")
priority = st.sidebar.radio("Manager priority", list(PRIORITY_LABELS),
                            format_func=PRIORITY_LABELS.get, index=1)

base_log, base_act = load_run("baseline")
log, act = load_run(priority)
day = st.sidebar.slider("Day", 0, int(log["day"].max()), 0)

tab_map, tab_ops, tab_results, tab_agent = st.tabs(
    ["🗺️ Map", "📋 Operations", "📊 Results", "🤖 Agent"])

# ---------- تبويب الخريطة ----------
# ---------- map tab ----------
with tab_map:
    b_day = base_log[base_log["day"] == day]
    n_day = log[log["day"] == day]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wasted today, without Ni'ma", int(b_day["wasted"].sum()))
    c2.metric("Wasted today, with Ni'ma", int(n_day["wasted"].sum()))
    c3.metric("Missed sales, without", int(b_day["stockout"].sum()))
    c4.metric("Missed sales, with", int(n_day["stockout"].sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Without Ni'ma")
        st.pydeck_chart(make_map(base_log, base_act, day))
    with right:
        st.subheader(f"With Ni'ma ({PRIORITY_LABELS[priority]})")
        st.pydeck_chart(make_map(log, act, day))
    st.caption("Circles grow and turn red with waste. Blue arcs are transfers between "
               "branches; green arcs are donations to food banks. Simulated data.")

# ---------- تبويب العمليات ----------
# ---------- operations tab ----------
with tab_ops:
    store = st.selectbox("Store", ["All stores"] + [s.name for s in sc.stores])
    rows = log[log["day"] == day]
    if store != "All stores":
        rows = rows[rows["store"] == store]
    st.subheader(f"Day {day}: stock and sales")
    st.dataframe(rows.drop(columns=["day"]), hide_index=True, width="stretch")

    st.subheader(f"Day {day}: tonight's actions")
    day_actions = act[act["day"] == day]
    if store != "All stores":
        day_actions = day_actions[day_actions["src"] == store]
    if day_actions.empty:
        st.info("No actions tonight.")
    else:
        st.dataframe(day_actions.drop(columns=["day"]), hide_index=True,
                     width="stretch")

# ---------- تبويب النتائج ----------
# ---------- results tab ----------
with tab_results:
    s = load_summary()
    b, m = s["baseline"], s[priority]
    st.subheader(f"30 days: {PRIORITY_LABELS[priority]} vs no redistribution")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Food wasted (units)", f"{m['wasted']:,}",
              f"{(m['wasted'] - b['wasted']) / b['wasted']:.0%}", delta_color="inverse")
    c2.metric("Missed sales (units)", f"{m['stockout']:,}",
              f"{(m['stockout'] - b['stockout']) / b['stockout']:.0%}", delta_color="inverse")
    c3.metric("Donated to food banks", f"{m['donated']:,}")
    c4.metric("Net revenue (SAR)", f"{m['net_revenue']:,.0f}",
              f"{m['net_revenue'] - b['net_revenue']:+,.0f}")

    # الهدر التراكمي عبر الأيام
    # cumulative waste over the days
    fig = go.Figure()
    for name, frame in [("Without Ni'ma", base_log), ("With Ni'ma", log)]:
        daily = frame.groupby("day")["wasted"].sum().cumsum()
        fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines", name=name))
    fig.update_layout(title="Cumulative food wasted", xaxis_title="Day",
                      yaxis_title="Units", height=380)
    st.plotly_chart(fig, width="stretch")

    st.subheader("All priorities compared")
    table = pd.DataFrame(s).T[["wasted", "stockout", "donated",
                               "revenue", "transport", "net_revenue"]]
    st.dataframe(table.style.format("{:,.0f}"), width="stretch")

# ---------- تبويب الوكيل ----------
# ---------- agent tab ----------
with tab_agent:
    events = load_events()
    st.subheader("Manager notes, and how Ni'ma responded")
    st.caption("The optimiser alone can't read notes. The agent reads them and adjusts the "
               "plan. Both faced the same customers and the same events.")

    brief_path = RUNS / "agent_briefings.csv"
    if not brief_path.exists():
        st.warning("Run `python run_all.py` to generate the agent's results.")
    else:
        briefs = pd.read_csv(brief_path)
        trans = pd.read_csv(RUNS / "agent_transcript.csv")
        solver_log, _ = load_run("balanced")
        agent_log, _ = load_run("agent")

        for e in events:
            f = e["effect"]
            days = f.get("days", [f.get("night")])

            # الهدر والمبيعات الضائعة خلال أيام الحدث، للمحسِّن وحده وللوكيل
            # waste and missed sales during the event, for the optimiser alone and for the agent
            def during(frame):
                w = frame[frame["day"].isin(days)]
                if e["store"] != "ALL":
                    w = w[w["store"] == e["store"]]
                return int(w["wasted"].sum()), int(w["stockout"].sum())

            s_waste, s_missed = during(solver_log)
            a_waste, a_missed = during(agent_log)
            with st.expander(f"Day {e['notice_day']} · {e['store']} · {e['text'][:70]}"):
                st.markdown(f"**Note:** {e['text']}")
                c1, c2 = st.columns(2)
                c1.metric("Wasted during the event (agent)", a_waste, a_waste - s_waste,
                          delta_color="inverse", help="Change compared with the optimiser alone")
                c2.metric("Missed sales during the event (agent)", a_missed, a_missed - s_missed,
                          delta_color="inverse", help="Change compared with the optimiser alone")
                b = briefs[briefs["day"] == e["notice_day"]]
                if not b.empty:
                    st.markdown("**Agent briefing**")
                    st.write(b["text"].iloc[0])
                t = trans[(trans["day"] == e["notice_day"]) & (trans["role"] == "tool")]
                if not t.empty:
                    st.markdown("**What the agent did**")
                    for line in t["content"]:
                        st.code(line, language=None)

    # ---------- العرض الحي ----------
    # ---------- live demo ----------
    st.divider()
    st.subheader("Try it live: send Ni'ma a note")
    names = [s.name for s in sc.stores]
    c1, c2 = st.columns([1, 3])
    note_store = c1.selectbox("From store", names, index=names.index("Rawdah"))
    note_day = c1.number_input("Evening of day", 0, 29, 11)
    note_text = c2.text_area(
        "Manager note (Arabic or English)",
        "عندنا عطل في ثلاجات فرع الروضة، الفني بيجي بكرة، الألبان بتخرب الليلة",
        height=110)
    if st.button("Send to Ni'ma", type="primary"):
        with st.spinner("Ni'ma is reading the note and planning..."):
            try:
                rows, actions, briefing, transcript = live_plan(
                    sc, note_store, note_text, int(note_day))
            except Exception as ex:
                st.error(f"Live run failed: {ex}")
            else:
                st.markdown("**Briefing**")
                st.write(briefing)
                acts = pd.DataFrame(actions, columns=ACTION_COLS)
                st.pydeck_chart(make_map(pd.DataFrame(rows), acts, int(note_day)))
                st.dataframe(acts.drop(columns=["day"]), hide_index=True, width="stretch")
                with st.expander("Agent steps"):
                    for t in transcript:
                        st.code(t["content"], language=None)