import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from sim.agent import (evaluate, live_night, store_actions, store_totals,
                       weekly_report)
from sim.models import load_scenario
from sim.simulator import ACTION_COLS

st.set_page_config(page_title="Ni'ma", page_icon="🌿", layout="wide")

RUNS = Path("runs")
INBOX = Path("data/inbox.json")
# أسماء السياسات كما يراها المدير
# policy names as the manager sees them
PRIORITY_LABELS = {"value": "Recover value (optimiser)",
                   "balanced": "Balanced (optimiser)",
                   "waste": "Reduce waste (optimiser)",
                   "agent": "Ni'ma AI agent"}
SHORT = {"value": "Recover value", "balanced": "Balanced", "waste": "Reduce waste"}
BLUE = [30, 120, 220, 200]
GREEN = [40, 170, 90, 200]
RED = [220, 60, 60, 200]
GREY = [120, 120, 120, 160]
TRUCK_COLORS = [[30, 120, 220], [230, 150, 30], [160, 80, 200],
                [40, 170, 90], [220, 60, 60], [0, 170, 170]]
ARABIC = re.compile(r"[\u0600-\u06FF]")
LATIN = re.compile(r"[A-Za-z]")
KIND_ICON = {"question": "❓ ", "alert": "⚠️ ", "approval_request": "✋ "}
MEASURES = {"Food wasted": "wasted", "Missed sales": "stockout",
            "Donated": "donated", "Net revenue": None}


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
NAMES = [s.name for s in sc.stores]
PRICE = {p.name: p.unit_value for p in sc.products}
KEEP = 1 - sc.discount_pct
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


def route_map(routes: list[dict], focus: int | None = None) -> pdk.Deck:
    """
    مسارات الشاحنات. عند اختيار شاحنة: نقطة البداية، والتوقفات المرقّمة، ونقطة النهاية.
    Truck routes. When one truck is selected: its start, numbered stops, and end.
    """
    stores = pd.DataFrame([{"position": [s.lon, s.lat], "store": s.name} for s in sc.stores])
    shown = [r for r in routes if focus is None or r["truck"] == focus]

    def color(r):
        # لون ثابت لكل شاحنة حتى لا يتغير بين العرضين
        # a fixed colour per truck, so it doesn't change between views
        return TRUCK_COLORS[(r["truck"] - 1) % len(TRUCK_COLORS)]

    layers = [
        pdk.Layer("ScatterplotLayer", data=stores, get_position="position",
                  get_radius=350, get_fill_color=GREY),
        pdk.Layer("TextLayer", data=stores, get_position="position", get_text="store",
                  get_size=14, get_color=[255, 255, 255],
                  get_alignment_baseline="'bottom'", get_pixel_offset=[0, -12]),
    ]
    if shown:
        layers.append(pdk.Layer(
            "PathLayer",
            data=pd.DataFrame([{"path": r["path"], "color": color(r),
                                "label": f"Truck {r['truck']}: " + " → ".join(r["names"])}
                               for r in shown]),
            get_path="path", get_color="color", get_width=8 if focus else 5,
            width_min_pixels=4 if focus else 3, pickable=True))

    if focus is not None and shown:
        r = shown[0]
        # بداية خضراء، ونهاية حمراء، وأرقام التوقفات بينهما
        # green start, red end, and stop numbers in between
        ends = pd.DataFrame([
            {"position": LOC[r["start"]], "color": [40, 170, 90], "label": f"Start: {r['start']}"},
            {"position": LOC[r["stops"][-1]["store"]], "color": [220, 60, 60],
             "label": f"End: {r['stops'][-1]['store']}"}])
        marks = [{"position": LOC[r["start"]], "text": "START", "color": [40, 170, 90]}]
        for k, s in enumerate(r["stops"], start=1):
            last = k == len(r["stops"])
            marks.append({"position": LOC[s["store"]],
                          "text": f"{k} · END" if last else str(k),
                          "color": [220, 60, 60] if last else color(r)})
        layers += [
            pdk.Layer("ScatterplotLayer", data=ends, get_position="position",
                      get_radius=700, get_fill_color="color", pickable=True),
            pdk.Layer("TextLayer", data=pd.DataFrame(marks), get_position="position",
                      get_text="text", get_size=18, get_color="color",
                      get_pixel_offset=[0, 20]),
        ]
    else:
        # في عرض كل الشاحنات: وسم صغير عند بداية كل شاحنة فقط
        # in the all-trucks view: just a small tag where each truck starts
        tags = pd.DataFrame([{"position": LOC[r["start"]], "text": f"T{r['truck']}",
                              "color": color(r)} for r in shown])
        if not tags.empty:
            layers.append(pdk.Layer("TextLayer", data=tags, get_position="position",
                                    get_text="text", get_size=18, get_color="color",
                                    get_pixel_offset=[0, 20]))

    return pdk.Deck(layers=layers,
                    initial_view_state=pdk.ViewState(latitude=24.69, longitude=46.72,
                                                     zoom=9.8, pitch=0),
                    map_style=None, tooltip={"text": "{label}"})


# ---------- عرض المحادثات ----------
# ---------- showing conversations ----------
def text_block(text: str, prefix: str = ""):
    """
    يعرض النص من اليمين لليسار إذا كان عربيًا في الغالب.
    Shows text right-to-left when it's mostly Arabic.
    """
    if len(ARABIC.findall(text)) > len(LATIN.findall(text)):
        st.markdown(f"<div dir='rtl' style='text-align:right'>{prefix}"
                    f"{html.escape(text)}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"{prefix}{text}")


def render_msg(m: dict):
    if m["role"] == "nima":
        with st.chat_message("Ni'ma", avatar="🌿"):
            text_block(m["text"], KIND_ICON.get(m.get("kind"), ""))
    elif m["role"] == "operations_manager":
        with st.chat_message("Operations", avatar="🧑‍💼"):
            text_block(m["text"])
    else:
        with st.chat_message("Manager", avatar="🏪"):
            text_block(m["text"])


def pending_questions(inbox: list[dict]) -> list[dict]:
    """
    أسئلة نعمة التي لم يُرد عليها بعد.
    Ni'ma's questions that haven't been answered yet.
    """
    pending = []
    for i, m in enumerate(inbox):
        if m["role"] == "nima" and m.get("kind") == "question":
            answered = any(x["role"] != "nima" and x["who"] == m["who"]
                           for x in inbox[i + 1:])
            if not answered:
                pending.append(m)
    return pending


def comparison_table(comp: dict, chosen: str) -> pd.DataFrame:
    """
    جدول المقارنة بين الخطط: ما تنقذه كل خطة، وإيرادها، وهل تحقق هدف المدير.
    The plan comparison table: what each plan saves, its revenue, and whether it meets the goal.
    """
    labels = {"ignore_notes": "If no one read the notes (not executable tonight)", **SHORT}
    rows = []
    for k in ["ignore_notes", "value", "balanced", "waste"]:
        if k in comp:
            r = comp[k]
            rows.append({"Option": labels[k] + ("  ✅ chosen" if k == chosen else ""),
                         "Trucks": r.get("trucks", 0),
                         "Transferred": r["transferred"], "Discounted": r["discounted"],
                         "Donated": r["donated"], "Units saved": r["expected_units_saved"],
                         "Net (SAR)": r["expected_net_sar"],
                         "Revenue given up": f"{r['revenue_given_up_pct']}%"
                         if "revenue_given_up_pct" in r else "",
                         "Meets goal": ("✅" if r.get("meets_goal") else "✗")
                         if "meets_goal" in r else ""})
    return pd.DataFrame(rows)


def daily_series(log: pd.DataFrame, actions: pd.DataFrame, measure: str) -> pd.Series:
    """
    القيمة التراكمية يومًا بيوم لمقياس واحد.
    The cumulative day-by-day value of one measure.
    """
    if measure == "Net revenue":
        pr = log["product"].map(PRICE)
        rev = (log["sold"] * pr + log["sold_discount"] * pr * KEEP).groupby(log["day"]).sum()
        cost = actions.groupby("day")["cost"].sum().reindex(rev.index, fill_value=0)
        return (rev - cost).cumsum()
    return log.groupby("day")[MEASURES[measure]].sum().cumsum()


# ---------- الشريط الجانبي ----------
# ---------- sidebar ----------
st.sidebar.title("Ni'ma  نعمة")
st.sidebar.caption("An AI agent that keeps food from going to waste across a grocery chain.")
priority = st.sidebar.radio("Policy", list(PRIORITY_LABELS),
                            format_func=PRIORITY_LABELS.get, index=3)

base_log, base_act = load_run("baseline")
log, act = load_run(priority)
LAST_DAY = int(base_log["day"].max())
day = st.sidebar.slider(
    "Day", 0, LAST_DAY, 11,
    help="The simulated day shown across the app. Ni'ma plans on this day's evening.")

tab_agent, tab_map, tab_ops, tab_results = st.tabs(
    ["🤖 Agent", "🗺️ Map", "📋 Operations", "📊 Results"])

# ---------- تبويب الوكيل ----------
# ---------- agent tab ----------
with tab_agent:
    if "inbox" not in st.session_state:
        st.session_state["inbox"] = json.loads(INBOX.read_text())

    st.subheader("Ni'ma's evening")
    st.caption("Store managers and the operations manager send notes. Ni'ma asks when "
               "something is unclear, chooses tonight's plan against the manager's goal, "
               "tells each person what to do, and reports to leadership.")

    # يوم التخطيط يأتي من شريط اليوم في الشريط الجانبي. آخر يوم لا يُخطَّط له لأنه لا يوجد غد
    # the planning day comes from the sidebar Day slider. The last day can't be planned: there's no tomorrow
    night = min(day, LAST_DAY - 1)
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
    c1.metric("Planning the evening of day", night)
    run = c2.button("▶ Run tonight's planning", type="primary", width="stretch")
    if c3.button("↺ Reset demo notes", width="stretch"):
        st.session_state["inbox"] = json.loads(INBOX.read_text())
        st.session_state.pop("night_result", None)
        st.rerun()

    if run:
        with st.spinner("Ni'ma is reading notes, comparing plans, checking roads "
                        "and messaging the team..."):
            try:
                res = live_night(sc, st.session_state["inbox"], night)
                # نحتفظ بأسئلة نعمة في المحادثة حتى تتبعها الإجابات
                # keep Ni'ma's questions in the conversation so answers can follow
                st.session_state["inbox"] += [m for m in res["messages"]
                                              if m["kind"] == "question"]
                st.session_state["night_result"] = res
                st.session_state["night_day"] = night
            except Exception as ex:
                st.error(f"Planning failed: {ex}")

    inbox = st.session_state["inbox"]
    res = st.session_state.get("night_result")
    if res and st.session_state.get("night_day") != night:
        st.info(f"The results below are for the evening of day "
                f"{st.session_state['night_day']}. Run the planning again to plan day {night}.")
    latest = [m for m in (res["messages"] if res else []) if m["kind"] != "question"]
    waiting = pending_questions(inbox)
    if waiting:
        st.warning(f"{len(waiting)} question(s) from Ni'ma waiting for an answer. "
                   "Reply below, then run the planning again.")

    t_store, t_opsmgr, t_report, t_week = st.tabs(
        ["🏪 Store managers", "🧑‍💼 Operations manager", "📑 Tonight's report",
         "📅 Weekly report"])

    # ١. طبقة مديري الفروع: نظرة على المتجر بجانب المحادثة
    # 1. store manager layer: the store overview next to the conversation
    with t_store:
        q_stores = {q["who"] for q in waiting}
        has_notes = [i for i, n in enumerate(NAMES) if any(m["who"] == n for m in inbox)]
        sel = st.selectbox("Store", NAMES, index=has_notes[0] if has_notes else 0,
                           format_func=lambda n: f"{n}  ❓ question waiting"
                           if n in q_stores else n)
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"#### 📦 {sel}: tonight at a glance")
            if not res:
                st.info("Run tonight's planning to see this store's overview.")
            else:
                r = pd.DataFrame(res["rows"])
                r = r[r["store"] == sel]
                exp = {e["product"]: e["qty"] for e in res["expiring_tomorrow"]
                       if e["store"] == sel}
                view = pd.DataFrame({
                    "Product": r["product"],
                    "Sold today": r["sold"],
                    "Wasted today": r["wasted"],
                    "Leaving tonight": r["transfer_out"] + r["donated"],
                    "Discounted": r["discounted"],
                    "Arriving tonight": r["transfer_in"],
                    "On shelf tomorrow": r["stock_end"],
                    "Expires tomorrow": r["product"].map(exp).fillna(0).astype(int),
                })
                st.dataframe(view, hide_index=True, width="stretch")
                lines = store_actions(res["plan"], sc).get(sel, [])
                st.markdown("**Ni'ma's instructions for tonight**")
                for line in lines or ["Nothing to do tonight."]:
                    st.markdown(f"- {line}")
        with right:
            st.markdown("#### 💬 Conversation with Ni'ma")
            thread = ([m for m in inbox if m["who"] == sel]
                      + [m for m in latest if m["who"] == sel])
            if not thread:
                st.caption("No messages yet.")
            for m in thread:
                render_msg(m)
            with st.form(f"reply_{sel}", clear_on_submit=True):
                txt = st.text_input(f"Write as the {sel} manager")
                if st.form_submit_button("Send") and txt:
                    st.session_state["inbox"].append(
                        {"role": "store_manager", "who": sel, "text": txt})
                    st.rerun()

    # ٢. طبقة مدير العمليات: المحادثة ومسارات الشاحنات مع التبرير
    # 2. operations manager layer: the conversation, and truck routes with justification
    with t_opsmgr:
        left, right = st.columns([2, 3])
        with left:
            st.markdown("#### 💬 Conversation with Ni'ma")
            for m in ([m for m in inbox if m["who"] == "Operations"]
                      + [m for m in latest if m["who"] == "Operations"]):
                render_msg(m)
            with st.form("reply_ops", clear_on_submit=True):
                txt = st.text_input("Write as the operations manager")
                if st.form_submit_button("Send") and txt:
                    st.session_state["inbox"].append(
                        {"role": "operations_manager", "who": "Operations", "text": txt})
                    st.rerun()
        with right:
            st.markdown("#### 🚚 Tonight's truck routes")
            if not res:
                st.info("Run tonight's planning to see the routes.")
            elif not res["routes"]:
                st.caption("No trucks needed tonight.")
            else:
                options = ["All trucks"] + [f"Truck {r['truck']}" for r in res["routes"]]
                choice = st.segmented_control("Show", options, default="All trucks",
                                              key="truck_pick")
                focus = (None if choice in (None, "All trucks")
                         else int(choice.split()[1]))
                st.pydeck_chart(route_map(res["routes"], focus))
                st.caption("Lines show the order of stops, not the exact roads. Simulated "
                           "traffic feed; in production, the Google Maps Routes API.")

                shown = [r for r in res["routes"] if focus is None or r["truck"] == focus]
                notes = (res.get("report") or {}).get("route_notes", [])
                if focus is not None:
                    notes = [n for n in notes if n.startswith(f"Truck {focus}")]
                    # الرحلة خطوة بخطوة
                    # the trip, step by step
                    r = shown[0]
                    st.markdown(f"**Start · {r['start']}**: load {r['units']} units")
                    for k, s in enumerate(r["stops"], start=1):
                        drop = ", ".join(f"{q} {p}" for p, q in s["items"])
                        road = "; ".join(s["incidents"]) or "road clear"
                        end = " · **end of route**" if k == len(r["stops"]) else ""
                        st.markdown(f"**Stop {k} · {s['store']}**: drop {drop} "
                                    f"({s['leg_km']} km, ~{s['leg_minutes']} min, {road}){end}")
                if notes:
                    st.markdown("**Why this route**" if focus else "**Why these routes**")
                    for line in notes:
                        st.markdown(f"- {line}")
                if focus is None:
                    for r in shown:
                        st.markdown(f"- **Truck {r['truck']}:** {' → '.join(r['names'])} · "
                                    f"{r['units']} units · {r['km']} km · ~{r['minutes']} min")

    # ٣. تقرير الليلة للإدارة العليا
    # 3. tonight's report for leadership
    with t_report:
        if not res:
            st.info("Run tonight's planning to see Ni'ma's report.")
        else:
            rep = res.get("report") or {}
            rday = st.session_state["night_day"]
            st.markdown(f"### {rep.get('headline') or res['summary']}")

            # أرقام التقرير محسوبة بالكود، لا يكتبها النموذج
            # the report's numbers are computed in code, not written by the model
            k = evaluate(res["plan"], sc)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Trucks", k["trucks"])
            m2.metric("Transferred", k["transferred"])
            m3.metric("Discounted", k["discounted"])
            m4.metric("Donated", k["donated"])
            m5.metric("Expected net (SAR)", f"{k['expected_net_sar']:,}")

            # قرار نعمة: أي خطة طُبّقت ولماذا، أو أمر مدير العمليات إن وُجد
            # Ni'ma's decision: which plan was applied and why, or the operations order if there was one
            dec = res.get("decision") or {}
            if dec:
                who = ("ordered by the operations manager" if dec.get("override")
                       else "Ni'ma's decision")
                st.info(f"**Plan applied: {SHORT[dec['priority']]}** ({who}). "
                        f"{dec.get('reason', '')}")

            left, right = st.columns(2)
            with left:
                st.markdown("#### What the notes changed")
                for line in rep.get("note_impacts", []):
                    st.markdown(f"- {line}")
                st.markdown("#### Adjustments applied")
                for line in res["adjustments"] or ["None"]:
                    st.markdown(f"- {line}")
            with right:
                st.markdown("#### Strategy")
                st.write(rep.get("strategy", ""))
                if res.get("comparison"):
                    st.dataframe(comparison_table(res["comparison"], res["priority"]),
                                 hide_index=True, width="stretch")
                    st.caption("'Meets goal': tonight's total revenue across the chain stays "
                               "within 1% of the best plan. 'If no one read the notes' ignores "
                               "tonight's spoilage, road closures, truck limit and food bank "
                               "availability, so its figures are not achievable.")

            st.markdown("#### Tonight's actions")
            acts = pd.DataFrame(res["actions"], columns=ACTION_COLS)
            st.pydeck_chart(make_map(pd.DataFrame(res["rows"]), acts, rday))

            st.markdown("#### What changed because of the notes")
            with_notes = store_totals(res["plan"], sc)
            without = store_totals(res["plan_without_notes"], sc)
            diff = [{"Store": n,
                     "Sends (ignore notes)": without[n]["sends"],
                     "Sends (Ni'ma)": with_notes[n]["sends"],
                     "Receives (ignore notes)": without[n]["receives"],
                     "Receives (Ni'ma)": with_notes[n]["receives"],
                     "Donates (ignore notes)": without[n]["donates"],
                     "Donates (Ni'ma)": with_notes[n]["donates"]}
                    for n in NAMES if any(with_notes[n].values()) or any(without[n].values())]
            if diff:
                st.dataframe(pd.DataFrame(diff), hide_index=True, width="stretch")

            st.markdown("#### Risks, open questions and approvals")
            risks = rep.get("risks", []) + [f"Waiting for {q['who']} to answer: {q['text']}"
                                            for q in waiting]
            for line in risks or ["None"]:
                st.markdown(f"- {line}")

            banks = [m for m in latest if m["to_role"] == "food_bank"]
            if banks:
                st.markdown("#### Messages to food banks")
                for m in banks:
                    with st.container(border=True):
                        st.markdown(f"**To: {m['who']}**")
                        text_block(m["text"])

            with st.expander("Ni'ma's step-by-step tool calls"):
                for t in res["transcript"]:
                    st.code(t["content"], language=None)

    # ٤. التقرير الأسبوعي للإدارة العليا
    # 4. the weekly report for leadership
    with t_week:
        st.caption("Ni'ma's weekly report for leadership. Figures are computed from the "
                   "30-day simulation, comparing Ni'ma with the chain without "
                   "redistribution; Ni'ma writes the narrative and recommendations.")
        weeks = [(s, min(s + 6, LAST_DAY)) for s in range(0, LAST_DAY + 1, 7)]
        current = next(i for i, (s, e) in enumerate(weeks) if s <= day <= e)
        w1, w2 = st.columns([2, 1], vertical_alignment="bottom")
        wk = w1.selectbox("Week", weeks, index=current,
                          format_func=lambda w: f"Week {w[0] // 7 + 1} (days {w[0]} to {w[1]})")
        write = w2.button("📝 Write weekly report", type="primary", width="stretch")

        if not (RUNS / "agent_log.csv").exists():
            st.warning("Run `python run_all.py` first, so there are results to report on.")
        elif write:
            with st.spinner("Ni'ma is reviewing the week..."):
                st.session_state["weekly"] = weekly_report(sc, *wk)

        wr = st.session_state.get("weekly")
        if wr:
            n, nar = wr["numbers"], wr["narrative"]
            st.markdown(f"### Week of days {n['days'].replace('-', ' to ')}")
            st.markdown(f"**{nar.get('headline', '')}**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Food wasted", f"{n['wasted']['with_nima']:,}",
                      f"{n['wasted']['change_pct']}% vs without", delta_color="inverse")
            k2.metric("Missed sales", f"{n['missed_sales']['with_nima']:,}",
                      f"{n['missed_sales']['change_pct']}% vs without", delta_color="inverse")
            k3.metric("Donated to food banks", f"{n['donated_units']:,}")
            k4.metric("Net revenue (SAR)", f"{n['net_revenue_sar']['with_nima']:,}",
                      f"{n['net_revenue_sar']['difference']:+,} vs without")

            # كم ليلة اختار نعمة كل خطة هذا الأسبوع
            # how many nights Ni'ma chose each plan this week
            if n.get("nights_by_plan"):
                st.markdown("**Ni'ma's nightly decisions this week:** " + ", ".join(
                    f"{SHORT.get(k, k)} on {v} night{'s' if v != 1 else ''}"
                    for k, v in n["nights_by_plan"].items()))

            h, i, rc = st.columns(3)
            for col, title, key in [(h, "Highlights", "highlights"),
                                    (i, "Issues", "issues"),
                                    (rc, "Recommendations", "recommendations")]:
                with col:
                    st.markdown(f"#### {title}")
                    for line in nar.get(key, []) or ["None"]:
                        st.markdown(f"- {line}")

            # الهدر لكل متجر هذا الأسبوع
            # waste per store this week
            ws = n["waste_by_store"]
            fig = go.Figure([
                go.Bar(name="Without Ni'ma", x=NAMES,
                       y=[ws["without"].get(s, 0) for s in NAMES]),
                go.Bar(name="With Ni'ma", x=NAMES,
                       y=[ws["with_nima"].get(s, 0) for s in NAMES]),
            ])
            fig.update_layout(barmode="group", title="Food wasted this week, by store",
                              yaxis_title="Units", height=360)
            st.plotly_chart(fig, width="stretch")

            if wr["notes"]:
                st.markdown("#### Notes Ni'ma handled this week")
                for e in wr["notes"]:
                    st.markdown(f"- **Day {e['day']} · {e['from']}:** {e['note']}")

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
    store = st.selectbox("Store", ["All stores"] + NAMES)
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
        st.dataframe(day_actions.drop(columns=["day"]), hide_index=True, width="stretch")

# ---------- تبويب النتائج ----------
# ---------- results tab ----------
with tab_results:
    s = load_summary()
    b, m = s["baseline"], s[priority]
    st.subheader(f"30 days: {PRIORITY_LABELS[priority]} vs no redistribution")
    st.caption("A 30-day simulation of six Riyadh stores. Every policy faces the same "
               "customers and the same events, so differences come only from decisions. "
               "'Without Ni'ma' is the chain as it works today: nothing is moved, "
               "discounted or donated.")

    v1, v2 = st.columns(2)
    view = v1.segmented_control("View", ["Compare", "Without Ni'ma", "With Ni'ma"],
                                default="Compare", key="res_view") or "Compare"
    measure = v2.segmented_control("Measure", list(MEASURES), default="Food wasted",
                                   key="res_measure") or "Food wasted"

    # الأرقام تتبع العرض المختار: بدون نعمة تعرض خط الأساس وحده
    # the numbers follow the chosen view: "Without Ni'ma" shows the baseline alone
    shown = b if view == "Without Ni'ma" else m
    with_delta = view != "Without Ni'ma"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Food wasted (units)", f"{shown['wasted']:,}",
              f"{(m['wasted'] - b['wasted']) / b['wasted']:.0%}" if with_delta else None,
              delta_color="inverse",
              help="Units thrown away because they passed their last sale day.")
    c2.metric("Missed sales (units)", f"{shown['stockout']:,}",
              f"{(m['stockout'] - b['stockout']) / b['stockout']:.0%}" if with_delta else None,
              delta_color="inverse",
              help="Customers who wanted a product, but the shelf was empty.")
    c3.metric("Donated to food banks", f"{shown['donated']:,}",
              help="Units given to food banks instead of thrown away.")
    c4.metric("Net revenue (SAR)", f"{shown['net_revenue']:,.0f}",
              f"{m['net_revenue'] - b['net_revenue']:+,.0f}" if with_delta else None,
              help="All sales, full price and discounted, minus transport. Deliveries are "
                   "identical across policies, so this difference is also the profit difference.")

    # نحسب الخطين دائمًا، حتى يبقى مقياس الرسم ثابتًا مهما كان العرض المختار
    # always compute both lines, so the chart's scale stays fixed whatever the view
    series = {"Without Ni'ma": daily_series(base_log, base_act, measure),
              "With Ni'ma": daily_series(log, act, measure)}
    top = max(float(v.max()) for v in series.values()) * 1.05
    visible = {"Compare": ["Without Ni'ma", "With Ni'ma"],
               "Without Ni'ma": ["Without Ni'ma"],
               "With Ni'ma": ["With Ni'ma"]}[view]

    fig = go.Figure()
    for name in visible:
        fig.add_trace(go.Scatter(x=series[name].index, y=series[name].values,
                                 mode="lines", name=name))
    unit = "SAR" if measure == "Net revenue" else "Units"
    fig.update_layout(title=f"Cumulative {measure.lower()} over 30 days",
                      xaxis_title="Day", yaxis_title=unit, height=420,
                      yaxis_range=[0, top], xaxis_range=[0, LAST_DAY])
    st.plotly_chart(fig, width="stretch")