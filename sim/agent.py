import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .models import Plan, distance_km
from .optimiser import PRIORITIES, plan_actions
from .policies import estimate
from .routes import build_routes

load_dotenv()

# النموذج ومفتاح الواجهة من ملف ‎.env
# model and API key come from the .env file
MODEL = os.getenv("NIMA_MODEL", "gpt-4.1-mini")
BASE_URL = os.getenv("NIMA_BASE_URL") or None
API_KEY = os.getenv("OPENAI_API_KEY", "")
# بعض النماذج لا تسمح بالأدوات إلا عند إيقاف التفكير المطوّل
# some models only allow tools when extended reasoning is switched off
REASONING_EFFORT = os.getenv("NIMA_REASONING_EFFORT", "none")
# أقصى عدد من جولات التفكير والأدوات في الليلة الواحدة
# maximum rounds of reasoning and tool calls per night
MAX_STEPS = 16

# الهدف الدائم الذي يضعه مدير العمليات: إنقاذ أكبر قدر من الطعام دون خسارة أكثر من 5% من أفضل إيراد
# the operations manager's standing goal: save the most food without giving up more than 5% of the best revenue
REVENUE_TOLERANCE = 0.05
DEFAULT_GOAL = (f"Save as much food as possible, as long as tonight's expected net revenue "
                f"stays within {REVENUE_TOLERANCE:.0%} of the best plan.")

SYSTEM_PROMPT = """You are Ni'ma, the operations agent for a grocery chain in Riyadh.
Stores: {stores}. Food banks: {food_banks}. Products: {products}.

Every evening you own one decision: what happens to food at risk of expiring, and which
plan to apply. An optimiser calculates three plans, one per priority: "value" (recover the
most money), "balanced", and "waste" (save the most food). You decide which one to apply.

The operations manager's standing goal: {goal}

Tonight is the evening of day {day}. Day {day}'s sales are over, so changes can only affect
day {tomorrow} or later.

You hear from two layers:
- Store managers report what is happening at their store: demand, events, equipment.
  Turn their notes into forecast changes with adjust_demand, close_store,
  cancel_delivery or mark_spoilage.
- The operations manager sets constraints: trucks, budget and food bank availability
  (set_truck_limit, set_transport_budget, block_food_bank). If they explicitly order a
  plan, call set_priority: that order overrides your own choice.

Trucks: each truck leaves one store and can drop at several stores on one route.

Work in this order:
1. Apply every clear note. Use realistic sizes: a large local event usually means 1.5 to
   2.5 times normal demand for the products mentioned; a competitor opening nearby
   usually means 0.6 to 0.9. If a note gives a number, use it.
2. If a note is too vague to act on (no products, no size, or no days), do not guess.
   Send that person a short question (send_message, kind "question") in their own
   language, and plan without that note for now. If the conversation already contains
   their answer, use it instead.
3. Call compare_plans. For each plan it reports the expected units saved, expected net
   revenue, the share of the best revenue it gives up, and whether it meets the standing
   goal. Choose the plan that saves the most food among those that meet the goal, unless
   the notes give a clear reason not to. Call choose_plan with a one-sentence reason
   citing those figures.
4. Call check_routes. If a truck must use a closed road, call block_route, then call
   compare_plans and choose_plan again, and alert the operations manager (kind "alert").
5. Send messages: one to the operations manager in English (kind "instruction")
   explaining the plan you chose and why, the truck routes, and anything that needs
   attention; one in Arabic to each store manager whose store sends, receives, discounts
   or donates anything, saying exactly what to do; one in Arabic to each food bank
   receiving a donation.
6. Call write_report for leadership. This ends the night.

Rules:
- Never state a figure you did not get from a tool.
- If a note refers to a day that has passed, tell whoever sent it.
- Never ask a question that has already been answered in the conversation.
- Keep messages short and practical, like real operations messages."""

ROUTINE_PROMPT = """You are Ni'ma, the operations agent for a grocery chain in Riyadh.
Every evening an optimiser calculates three plans: "value" (recover the most money),
"balanced", and "waste" (save the most food). You decide which one to apply.

The operations manager's standing goal: {goal}

Tonight is the evening of day {day}, and there are no notes from the team.
Call compare_plans. Then call choose_plan with the plan that saves the most food among
those that meet the standing goal, and a one-sentence reason citing the figures. Then stop.
Never state a figure you did not get from a tool."""


# ---------- أدوات مساعدة لوصف الخطة ----------
# ---------- helpers that describe a plan ----------
def evaluate(plan: Plan, sc) -> dict:
    """
    النتائج المتوقعة للخطة: وحدات مُنقذة، وإيراد، وتكلفة نقل. تُحسب بالكود وليس بالنموذج.
    A plan's expected results: units saved, revenue, transport cost. Computed in code, not by the model.
    """
    value = {p.name: p.unit_value for p in sc.products}
    q, keep = sc.discount_sell_prob, 1 - sc.discount_pct
    trucks = {t.src for t in plan.transfers}
    transport = (sum(t.qty * distance_km(sc.store(t.src), sc.store(t.dst))
                     * sc.transport_cost_per_unit_km for t in plan.transfers)
                 + sc.trip_cost * len(trucks))
    revenue = (sum(t.qty * value[t.product] for t in plan.transfers)
               + sum(q * keep * c.qty * value[c.product] for c in plan.discounts))
    saved = (sum(t.qty for t in plan.transfers)
             + q * sum(c.qty for c in plan.discounts)
             + sum(g.qty for g in plan.donations))
    return {
        "trucks": len(trucks),
        "transferred": sum(t.qty for t in plan.transfers),
        "discounted": sum(c.qty for c in plan.discounts),
        "donated": sum(g.qty for g in plan.donations),
        "expected_units_saved": round(saved),
        "expected_revenue_sar": round(revenue),
        "transport_sar": round(transport),
        "expected_net_sar": round(revenue - transport),
    }


def store_actions(plan: Plan, sc) -> dict:
    """
    ما يجب أن يفعله كل متجر وكل جهة مستفيدة الليلة، لكتابة الرسائل بدقة.
    What each store and food bank must do tonight, so messages are accurate.
    """
    out: dict = {}

    def add(who, line):
        out.setdefault(who, []).append(line)

    pct = round(sc.discount_pct * 100)
    for t in plan.transfers:
        add(t.src, f"send {t.qty} {t.product} to {t.dst}")
        add(t.dst, f"receive {t.qty} {t.product} from {t.src}")
    for c in plan.discounts:
        add(c.store, f"discount {c.qty} {c.product} by {pct}%")
    for g in plan.donations:
        add(g.store, f"hand {g.qty} {g.product} to {g.recipient}")
        add(g.recipient, f"collect {g.qty} {g.product} from {g.store}")
    return out


def store_totals(plan: Plan, sc) -> dict:
    """
    مجاميع لكل متجر: ما يستقبله ويرسله ويخفّضه ويتبرع به.
    Totals per store: what it receives, sends, discounts and donates.
    """
    tot = {s.name: {"receives": 0, "sends": 0, "discounts": 0, "donates": 0}
           for s in sc.stores}
    for t in plan.transfers:
        tot[t.dst]["receives"] += t.qty
        tot[t.src]["sends"] += t.qty
    for c in plan.discounts:
        tot[c.store]["discounts"] += c.qty
    for g in plan.donations:
        tot[g.store]["donates"] += g.qty
    return tot


def format_conversation(thread: list[dict]) -> str:
    """
    يحوّل المحادثة إلى نص يقرؤه النموذج، الأقدم أولًا.
    Turns the conversation into text for the model, oldest first.
    """
    lines = []
    for m in thread:
        if m["role"] == "nima":
            lines.append(f"[Ni'ma -> {m['who']}] ({m.get('kind', 'message')}) {m['text']}")
        elif m["role"] == "operations_manager":
            lines.append(f"[Operations manager] {m['text']}")
        else:
            lines.append(f"[Store manager, {m['who']}] {m['text']}")
    return "\n".join(lines)


def tool_schemas(sc, routine: bool = False) -> list[dict]:
    """
    تعريفات الأدوات التي يراها النموذج. في الليالي الروتينية: المقارنة والاختيار فقط.
    The tool definitions the model sees. On routine nights: only comparing and choosing.
    """
    stores = [s.name for s in sc.stores]
    banks = [r.name for r in sc.recipients]
    products = [p.name for p in sc.products] + ["all"]
    store = {"type": "string", "enum": stores}
    prods = {"type": "array", "items": {"type": "string", "enum": products}}
    days = {"type": "array", "items": {"type": "integer"}}
    text = {"type": "string"}
    priority = {"type": "string", "enum": list(PRIORITIES)}

    def fn(name, description, props, required):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}

    decide = [
        fn("compare_plans", "Compare the three plans: expected units saved, expected net "
           "revenue, the share of the best revenue each gives up, and whether each meets "
           "the standing goal.", {}, []),
        fn("choose_plan", "Apply one plan tonight. Returns the plan and what each store "
           "must do.", {"priority": priority, "reason": text}, ["priority", "reason"]),
    ]
    if routine:
        return decide

    return decide + [
        # أدوات طبقة مديري الفروع
        # store manager layer
        fn("get_overview", "Tonight's surplus (food that will expire unsold) and need "
           "(expected shortfall) for every store.", {}, []),
        fn("adjust_demand", "Expected demand at a store will change on the given days.",
           {"store": store, "products": prods, "multiplier": {"type": "number"}, "days": days},
           ["store", "products", "multiplier", "days"]),
        fn("close_store", "A store will make no sales on the given days.",
           {"store": store, "days": days}, ["store", "days"]),
        fn("cancel_delivery", "Warehouse deliveries of these products will not arrive "
           "on the given days, at any store.",
           {"products": prods, "days": days}, ["products", "days"]),
        fn("mark_spoilage", "Stock of these products at this store will be lost tonight.",
           {"store": store, "products": prods}, ["store", "products"]),
        # أدوات طبقة مدير العمليات
        # operations manager layer
        fn("set_priority", "Only when the operations manager explicitly orders a plan. "
           "This overrides your own choice tonight.",
           {"priority": priority, "reason": text}, ["priority", "reason"]),
        fn("set_truck_limit", "At most this many trucks can run tonight.",
           {"trucks": {"type": "integer"}}, ["trucks"]),
        fn("set_transport_budget", "Tonight's total transport cost must stay within this "
           "many SAR.", {"sar": {"type": "number"}}, ["sar"]),
        fn("block_food_bank", "This food bank can't receive donations tonight.",
           {"food_bank": {"type": "string", "enum": banks}}, ["food_bank"]),
        # الطرق
        # roads
        fn("check_routes", "Build tonight's truck routes and check live traffic on every "
           "leg: road distance, travel time, closures and delays.", {}, []),
        fn("block_route", "Avoid sending stock between these two stores tonight.",
           {"src": store, "dst": store, "reason": text}, ["src", "dst", "reason"]),
        # التواصل والتقرير
        # communication and reporting
        fn("send_message", "Send a message to one person.",
           {"to_role": {"type": "string",
                        "enum": ["operations_manager", "store_manager", "food_bank"]},
            "to": {"type": "string",
                   "description": "Store name, food bank name, or 'Operations'"},
            "language": {"type": "string", "enum": ["en", "ar"]},
            "kind": {"type": "string",
                     "enum": ["instruction", "question", "alert", "approval_request"]},
            "text": text},
           ["to_role", "to", "language", "kind", "text"]),
        fn("write_report", "Write tonight's report for leadership. This ends the night.",
           {"headline": {"type": "string", "description": "Two sentences: tonight's decision"},
            "note_impacts": {"type": "array", "items": text,
                             "description": "Each note and what it changed"},
            "strategy": {"type": "string",
                         "description": "Plans compared, which one you chose and why"},
            "route_notes": {"type": "array", "items": text,
                            "description": "One sentence per truck, starting 'Truck N:', "
                                           "explaining why it takes this route and these stops"},
            "risks": {"type": "array", "items": text,
                      "description": "Risks, open questions and approvals pending"}},
           ["headline", "note_impacts", "strategy", "route_notes", "risks"]),
    ]


class NimaAgent:
    """
    وكيل نعمة: يستمع لمديري الفروع ومدير العمليات، ويقرر كل ليلة أي خطة يطبّق، وينفّذ، ويراسل، ويرفع تقريرًا.
    Ni'ma agent: listens to store and operations managers, decides every night which plan
    to apply, executes, messages each role, and reports upward.
    """

    def __init__(self, priority: str = "balanced", use_llm: bool = True,
                 goal: str = DEFAULT_GOAL):
        # الخطة الافتراضية إذا تعذّر على النموذج أن يقرر
        # the default plan if the model can't decide
        self.default_priority = priority
        self.priority = priority
        self.goal = goal
        self.use_llm = use_llm
        # تعديلات الطلب تبقى سارية للأيام المحددة
        # demand adjustments stay in force for their days
        self.demand_events: list[dict] = []
        # التوريدات الملغاة: (المنتج، اليوم)
        # cancelled deliveries: (product, day)
        self.no_delivery: set = set()
        # المحادثة القادمة من التطبيق: ملاحظات وردود وأسئلة سابقة
        # the conversation from the app: notes, replies and earlier questions
        self.inbox: list[dict] = []
        # سجل قرارات نعمة كل ليلة وأسبابها
        # log of Ni'ma's decision each night, and why
        self.decisions: list[dict] = []
        self.transcript: list[dict] = []
        self.last_plan: Plan | None = None
        self.last_result: dict = {}
        self._client = None
        self._reset_night()

    def _reset_night(self):
        """
        قيود الليلة الواحدة تبدأ من جديد كل مساء.
        Single-night constraints start fresh every evening.
        """
        self.priority = self.default_priority
        self.override: str | None = None
        self.spoil: set = set()
        self.truck_limit: int | None = None
        self.budget: float | None = None
        self.blocked_routes: set = set()
        self.blocked_recipients: set = set()
        self.night_log: list[str] = []

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        return self._client

    # ---------- مدخلات المحسِّن بعد تعديلات الوكيل ----------
    # ---------- optimiser inputs after the agent's adjustments ----------
    def _demand_adj(self, sim) -> dict:
        """
        متوسط معامل الطلب على أيام الأفق القادمة لكل متجر ومنتج.
        The average demand multiplier over the coming horizon for each store and product.
        """
        d, h = sim.day, sim.sc.delivery_every
        adj = {}
        for s in sim.sc.stores:
            for p in sim.sc.products:
                factors = []
                for t in range(d + 1, d + h + 1):
                    f = 1.0
                    for e in self.demand_events:
                        if e["store"] == s.name and t in e["days"] and p.name in e["products"]:
                            f *= e["multiplier"]
                    factors.append(f)
                m = sum(factors) / len(factors)
                if m != 1.0:
                    adj[(s.name, p.name)] = m
        return adj

    def _inputs(self, sim):
        return estimate(sim, demand_adj=self._demand_adj(sim),
                        no_delivery=self.no_delivery, spoil=self.spoil)

    def _plan(self, sim, priority: str | None = None) -> Plan:
        surplus, urgent, need = self._inputs(sim)
        return plan_actions(sim.sc, surplus, need, priority or self.priority,
                            urgent=urgent, no_discount=self.spoil,
                            max_trips=self.truck_limit, budget=self.budget,
                            blocked_routes=self.blocked_routes,
                            blocked_recipients=self.blocked_recipients)

    def _plan_ignoring_notes(self, sim) -> Plan:
        surplus, urgent, need = estimate(sim)
        return plan_actions(sim.sc, surplus, need, self.default_priority, urgent=urgent)

    def _overview(self, sim) -> dict:
        surplus, _, need = self._inputs(sim)
        return {s.name: {
            "surplus": {p: v for (st, p), v in surplus.items() if st == s.name and v > 0},
            "need": {p: v for (st, p), v in need.items() if st == s.name and v > 0},
        } for s in sim.sc.stores}

    def _compare(self, sim, state) -> dict:
        """
        يقارن الخطط الثلاث. الكود يحسب نسبة الإيراد المتنازل عنها، وهل تحقق كل خطة هدف المدير.
        Compares the three plans. The code calculates the share of revenue given up, and
        whether each plan meets the manager's goal.
        """
        out = {pr: evaluate(self._plan(sim, pr), sim.sc) for pr in PRIORITIES}
        best = max(r["expected_net_sar"] for r in out.values())
        for r in out.values():
            given = best - r["expected_net_sar"]
            r["revenue_given_up_pct"] = round(given / best * 100, 1) if best > 0 else 0.0
            r["meets_goal"] = given <= REVENUE_TOLERANCE * max(best, 0)
        if state.get("plan_without_notes") is not None:
            out["ignore_notes"] = evaluate(state["plan_without_notes"], sim.sc)
        state["comparison"] = out
        return out

    @staticmethod
    def _expand(products: list, sc) -> list[str]:
        names = [p.name for p in sc.products]
        return names if "all" in products else [p for p in products if p in names]

    # ---------- تنفيذ الأدوات ----------
    # ---------- tool execution ----------
    def _call_tool(self, sim, name: str, args: dict, state: dict) -> dict:
        sc = sim.sc
        stores = [s.name for s in sc.stores]
        for key in ("store", "src", "dst"):
            if key in args and args[key] not in stores:
                return {"error": f"unknown store {args[key]}"}

        # لا يمكن تغيير يوم انتهى: نطبّق على الأيام القادمة فقط ونبلّغ النموذج
        # a day that has ended can't change: apply only to future days and tell the model
        all_days = [int(x) for x in args.get("days", [])]
        days = [t for t in all_days if t > sim.day]
        past = [t for t in all_days if t <= sim.day]
        warning = ({"warning": f"days {past} have already ended and were ignored"}
                   if past else {})
        if past and name in ("adjust_demand", "close_store", "cancel_delivery"):
            self.night_log.append(
                f"Ignored day {', '.join(map(str, past))}: that day is already over")
        day_text = ", ".join(map(str, days))

        # أي تعديل يجعل الخطة الحالية قديمة حتى يختار الوكيل من جديد
        # any change makes the current plan stale until the agent chooses again
        changes = {"adjust_demand", "close_store", "cancel_delivery", "mark_spoilage",
                   "set_priority", "set_truck_limit", "set_transport_budget",
                   "block_food_bank", "block_route"}
        if name in changes:
            state["stale"] = True

        if name == "get_overview":
            return self._overview(sim)
        if name == "adjust_demand":
            # حدود آمنة للمعامل، حتى لا يبالغ النموذج في التوقعات
            # safe limits on the multiplier, so the model can't overreact
            m = min(3.0, max(0.3, float(args.get("multiplier", 1.0))))
            prods = self._expand(args.get("products", []), sc)
            if days:
                self.demand_events.append({"store": args["store"], "multiplier": m,
                                           "days": days, "products": prods})
                self.night_log.append(f"{args['store']}: expected demand ×{m:g} for "
                                      f"{', '.join(prods)} on day {day_text}")
            return {"ok": True, "applied_multiplier": m, "applied_days": days, **warning}
        if name == "close_store":
            if days:
                self.demand_events.append({"store": args["store"], "multiplier": 0.0,
                                           "days": days, "products": self._expand(["all"], sc)})
                self.night_log.append(f"{args['store']}: closed on day {day_text}, no sales")
            return {"ok": True, "applied_days": days, **warning}
        if name == "cancel_delivery":
            prods = self._expand(args.get("products", []), sc)
            for p in prods:
                for t in days:
                    self.no_delivery.add((p, t))
            if days:
                self.night_log.append(f"No warehouse delivery of {', '.join(prods)} "
                                      f"on day {day_text}")
            return {"ok": True, "applied_days": days, **warning}
        if name == "mark_spoilage":
            prods = self._expand(args.get("products", []), sc)
            for p in prods:
                self.spoil.add((args["store"], p))
            self.night_log.append(f"{args['store']}: {', '.join(prods)} will spoil tonight, "
                                  "so all of it must leave the store")
            return {"ok": True}
        if name == "set_priority":
            if args.get("priority") not in PRIORITIES:
                return {"error": "unknown priority"}
            self.override = args["priority"]
            self.night_log.append(f"Operations ordered the '{self.override}' plan "
                                  f"({args.get('reason', '')})")
            return {"ok": True, "note": "choose_plan must now use this priority"}
        if name == "set_truck_limit":
            self.truck_limit = max(0, int(args.get("trucks", 0)))
            self.night_log.append(f"Operations: at most {self.truck_limit} trucks tonight")
            return {"ok": True}
        if name == "set_transport_budget":
            self.budget = max(0.0, float(args.get("sar", 0)))
            self.night_log.append(f"Operations: transport budget {self.budget:,.0f} SAR")
            return {"ok": True}
        if name == "block_food_bank":
            bank = args.get("food_bank", "")
            if bank not in [r.name for r in sc.recipients]:
                return {"error": f"unknown food bank {bank}"}
            self.blocked_recipients.add(bank)
            self.night_log.append(f"Operations: no donations to {bank} tonight")
            return {"ok": True}
        if name == "compare_plans":
            return self._compare(sim, state)
        if name == "choose_plan":
            pr = args.get("priority")
            if pr not in PRIORITIES:
                return {"error": "unknown priority"}
            # أمر مدير العمليات الصريح يعلو على اختيار الوكيل، والكود يفرض ذلك
            # an explicit operations order outranks the agent's choice, and the code enforces it
            if self.override and pr != self.override:
                return {"error": f"the operations manager ordered the '{self.override}' plan "
                                 "tonight; choose it"}
            self.priority = pr
            reason = args.get("reason", "")
            state["decision"] = {"priority": pr, "reason": reason,
                                 "override": bool(self.override)}
            state["plan"] = self._plan(sim)
            state["stale"] = False
            self.night_log.append(f"Ni'ma chose the '{pr}' plan: {reason}")
            return {**evaluate(state["plan"], sc), "priority": pr,
                    "by_store": store_actions(state["plan"], sc)}
        if name == "check_routes":
            plan = state["plan"] or self._plan(sim)
            return {"trucks": [{
                "truck": r["truck"], "route": " -> ".join(r["names"]),
                "units": r["units"], "km": r["km"], "minutes": r["minutes"],
                "closed_legs": r["closed_legs"],
                "incidents": sorted({i for s in r["stops"] for i in s["incidents"]}),
            } for r in build_routes(plan, sc)]}
        if name == "block_route":
            a, b = args["src"], args["dst"]
            self.blocked_routes |= {(a, b), (b, a)}
            self.night_log.append(f"Route {a} ↔ {b} avoided: {args.get('reason', '')}")
            return {"ok": True}
        if name == "send_message":
            to_role = args.get("to_role", "operations_manager")
            to = args.get("to", "")
            # نطابق الاسم مع متجر أو جهة مستفيدة معروفة
            # match the name to a known store or food bank
            known = stores + [r.name for r in sc.recipients]
            who = "Operations" if to_role == "operations_manager" else next(
                (k for k in known if k.lower() in to.lower()), to)
            state["messages"].append({"role": "nima", "to_role": to_role, "who": who,
                                      "language": args.get("language", "en"),
                                      "kind": args.get("kind", "instruction"),
                                      "text": args.get("text", "")})
            return {"ok": True}
        if name == "write_report":
            state["report"] = {"headline": args.get("headline", ""),
                               "note_impacts": args.get("note_impacts", []),
                               "strategy": args.get("strategy", ""),
                               "route_notes": args.get("route_notes", []),
                               "risks": args.get("risks", [])}
            return {"ok": True}
        return {"error": f"unknown tool {name}"}

    # ---------- حلقة الوكيل ----------
    # ---------- the agent loop ----------
    def _run_llm(self, sim, thread: list[dict], routine: bool) -> Plan:
        sc = sim.sc
        fields = dict(stores=", ".join(s.name for s in sc.stores),
                      food_banks=", ".join(r.name for r in sc.recipients),
                      products=", ".join(p.name for p in sc.products),
                      day=sim.day, tomorrow=sim.day + 1, goal=self.goal)
        system = (ROUTINE_PROMPT if routine else SYSTEM_PROMPT).format(**fields)
        user = ("No notes tonight." if routine else
                "Tonight's conversation with the team, oldest first:\n"
                + format_conversation(thread))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.transcript.append({"day": sim.day, "role": "notes", "content": user})
        state = {"plan": None, "stale": False, "report": None, "messages": [],
                 "comparison": None, "decision": None,
                 "plan_without_notes": None if routine else self._plan_ignoring_notes(sim)}
        reply = ""
        extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
        tools = tool_schemas(sc, routine)

        for _ in range(MAX_STEPS):
            resp = self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, tool_choice="auto", **extra)
            msg = resp.choices[0].message
            reply = msg.content or ""
            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": reply})
                break
            # نعيد إرسال طلبات الأدوات بالحقول التي تقبلها الواجهة فقط
            # send the tool calls back with only the fields the API accepts
            messages.append({"role": "assistant", "content": reply,
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._call_tool(sim, tc.function.name, args, state)
                self.transcript.append({
                    "day": sim.day, "role": "tool",
                    "content": f"{tc.function.name}({json.dumps(args, ensure_ascii=False)})"
                               f" -> {json.dumps(result, ensure_ascii=False)[:400]}"})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
            if (routine and state["decision"]) or (not routine and state["report"]):
                break

        # الخطة النهائية تحترم كل القيود: إن تغيّر شيء بعد آخر اختيار نعيد التشغيل
        # the final plan respects every constraint: if anything changed since the last choice, rerun
        plan = state["plan"] if state["plan"] and not state["stale"] else self._plan(sim)
        decision = state["decision"] or {
            "priority": self.priority, "override": bool(self.override),
            "reason": "No choice was made, so the default plan was applied."}
        report = state["report"]
        if routine:
            plan.explanation = (f"Routine night. Ni'ma chose the '{decision['priority']}' "
                                f"plan: {decision['reason']}")
        else:
            plan.explanation = (report["headline"] if report else "") or reply or \
                "Plan created by the optimiser."
        self.last_result = {"summary": plan.explanation, "report": report,
                            "messages": state["messages"], "comparison": state["comparison"],
                            "plan_without_notes": state["plan_without_notes"] or plan,
                            "adjustments": list(self.night_log), "priority": self.priority,
                            "base_priority": self.default_priority, "decision": decision,
                            # المسارات تُبنى من الخطة النهائية بالكود، فلا تصبح قديمة
                            # routes are built from the final plan in code, so they can't go stale
                            "routes": build_routes(plan, sc)}
        return plan

    def __call__(self, sim) -> Plan:
        """
        يُستدعى كل مساء. الوكيل يقرر كل ليلة؛ وفي الليالي التي فيها ملاحظات يدير الليلة كاملة.
        Called every evening. The agent decides every night; on nights with notes it runs
        the full evening.
        """
        self._reset_night()
        # ملاحظات الأحداث المجدولة، ثم المحادثة القادمة من التطبيق
        # notes from scheduled events, then the conversation from the app
        thread = [{"role": "operations_manager" if n["store"] == "ALL" else "store_manager",
                   "who": n["store"], "text": n["text"]} for n in sim.notes_today()]
        thread += self.inbox
        self.inbox = []
        has_notes = any(m["role"] != "nima" for m in thread)

        if self.use_llm:
            try:
                plan = self._run_llm(sim, thread, routine=not has_notes)
            except Exception as e:
                # عند تعطل النموذج: نستخدم خطة المحسِّن الافتراضية ونبلّغ بذلك
                # if the model fails: use the optimiser's default plan and say so
                plan = self._plan(sim)
                plan.explanation = (f"AI agent unavailable tonight ({type(e).__name__}: "
                                    f"{str(e)[:300]}). The default plan was applied.")
                self.last_result = self._plain_result(plan, sim.sc)
        else:
            plan = self._plan(sim)
            plan.explanation = "Optimiser only: the default plan was applied."
            self.last_result = self._plain_result(plan, sim.sc)

        d = self.last_result["decision"]
        self.decisions.append({"day": sim.day, "priority": d["priority"],
                               "reason": d["reason"], "notes": has_notes})
        self.last_plan = plan
        return plan

    def _plain_result(self, plan: Plan, sc) -> dict:
        return {"summary": plan.explanation, "report": None, "messages": [],
                "comparison": None, "plan_without_notes": plan,
                "adjustments": list(self.night_log), "priority": self.priority,
                "base_priority": self.default_priority,
                "decision": {"priority": self.priority, "override": False,
                             "reason": "Default plan (no decision by the agent)."},
                "routes": build_routes(plan, sc)}


def live_night(sc, inbox: list[dict], day: int, priority: str = "balanced",
               seed: int = 0) -> dict:
    """
    للعرض الحي: يشغّل المحاكاة حتى اليوم المحدد، ثم يدير الوكيل ليلة كاملة بناءً على محادثة الفريق.
    For the live demo: runs the simulation up to a day, then the agent runs a full night
    from the team's conversation.
    """
    from .policies import make_solver_policy
    from .simulator import Simulation

    sim = Simulation(sc, seed)
    warmup = make_solver_policy(priority)
    for _ in range(day):
        sim.step(warmup)
    agent = NimaAgent(priority)
    agent.inbox = list(inbox)
    rows, actions, _ = sim.step(agent)

    # ما ينتهي غدًا في كل متجر بعد تنفيذ خطة الليلة
    # what expires tomorrow at each store after tonight's plan
    expiring: dict = {}
    for b in sim.batches:
        if b.expiry_day <= sim.day:
            expiring[(b.store, b.product)] = expiring.get((b.store, b.product), 0) + b.qty
    return {"rows": rows, "actions": actions, "plan": agent.last_plan,
            "transcript": agent.transcript,
            "expiring_tomorrow": [{"store": s, "product": p, "qty": q}
                                  for (s, p), q in expiring.items()],
            **agent.last_result}


# ---------- التقرير الأسبوعي ----------
# ---------- the weekly report ----------
WEEKLY_PROMPT = """You are Ni'ma, the operations agent for a grocery chain in Riyadh.
Write this week's report for the chain's leadership. You receive the standing goal, the
week's figures (computed by the system by comparing Ni'ma with the chain operating
without redistribution), the manager notes received, your nightly briefings, and your
nightly decisions about which plan to apply.

Return JSON with these keys:
- "headline": two sentences on the week's outcome.
- "highlights": 3 to 5 short strings: the biggest results, the notable nights, and how
  your nightly plan choices served the standing goal.
- "issues": 2 to 4 short strings: where waste or missed sales remain, and why.
- "recommendations": 2 to 4 short, concrete strings for next week: stocking,
  delivery or operations changes.

Rules: use only the numbers you are given; never invent figures. Plain, professional
English."""


def weekly_report(sc, start: int, end: int, runs_dir: str = "runs",
                  events_path: str = "data/events.json") -> dict:
    """
    التقرير الأسبوعي للإدارة العليا: الأرقام تُحسب بالكود من نتائج المحاكاة، والنموذج يكتب السرد والتوصيات فقط.
    Weekly report for leadership: numbers are computed in code from the simulation results;
    the model only writes the narrative and recommendations.
    """
    from pathlib import Path

    import pandas as pd

    from .metrics import summarise

    runs = Path(runs_dir)
    days = list(range(start, end + 1))

    def load(name):
        log = pd.read_csv(runs / f"{name}_log.csv")
        act = pd.read_csv(runs / f"{name}_actions.csv")
        return log[log["day"].isin(days)], act[act["day"].isin(days)]

    a_log, a_act = load("agent")
    b_log, b_act = load("baseline")
    a, b = summarise(sc, a_log, a_act), summarise(sc, b_log, b_act)

    def pct(x, y):
        return round((x - y) / y * 100) if y else 0

    def by(frame, col):
        return {k: int(v) for k, v in frame.groupby(col)["wasted"].sum().items()}

    # كل الأرقام هنا محسوبة بالكود
    # every number here is computed in code
    numbers = {
        "days": f"{start}-{end}",
        "wasted": {"with_nima": a["wasted"], "without": b["wasted"],
                   "change_pct": pct(a["wasted"], b["wasted"])},
        "missed_sales": {"with_nima": a["stockout"], "without": b["stockout"],
                         "change_pct": pct(a["stockout"], b["stockout"])},
        "donated_units": a["donated"],
        "net_revenue_sar": {"with_nima": round(a["net_revenue"]),
                            "without": round(b["net_revenue"]),
                            "difference": round(a["net_revenue"] - b["net_revenue"])},
        "transport_sar": round(a["transport"]),
        "units_by_action": {t: int(a_act.loc[a_act["type"] == t, "qty"].sum())
                            for t in ["transfer", "discount", "donation"]},
        "waste_by_product": {"with_nima": by(a_log, "product"),
                             "without": by(b_log, "product")},
        "waste_by_store": {"with_nima": by(a_log, "store"),
                           "without": by(b_log, "store")},
    }

    decisions = []
    dec_path = runs / "agent_decisions.csv"
    if dec_path.exists():
        ddf = pd.read_csv(dec_path)
        ddf = ddf[ddf["day"].isin(days)]
        numbers["nights_by_plan"] = {k: int(v) for k, v in ddf["priority"].value_counts().items()}
        decisions = [{"day": int(r.day), "plan": r.priority, "reason": r.reason}
                     for r in ddf.itertuples()]

    events = (json.loads(Path(events_path).read_text())
              if Path(events_path).exists() else [])
    notes = [{"day": e["notice_day"], "from": e["store"], "note": e["text"]}
             for e in events if start <= e["notice_day"] <= end]

    briefs = []
    brief_path = runs / "agent_briefings.csv"
    if brief_path.exists():
        bdf = pd.read_csv(brief_path)
        bdf = bdf[bdf["day"].isin(days) & ~bdf["text"].str.startswith("Routine night")]
        briefs = [{"day": int(r.day), "briefing": r.text} for r in bdf.itertuples()]

    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": WEEKLY_PROMPT},
                      {"role": "user", "content": json.dumps(
                          {"standing_goal": DEFAULT_GOAL, "figures": numbers,
                           "notes": notes, "nightly_briefings": briefs,
                           "nightly_decisions": decisions}, ensure_ascii=False)}],
            response_format={"type": "json_object"}, **extra)
        narrative = json.loads(resp.choices[0].message.content)
    except Exception as e:
        # عند تعطل النموذج: تبقى الأرقام صحيحة لأنها محسوبة بالكود
        # if the model fails: the figures stay correct, since they're computed in code
        narrative = {"headline": f"Weekly narrative unavailable ({type(e).__name__}). "
                                 "The figures below are computed directly from the simulation.",
                     "highlights": [], "issues": [], "recommendations": []}
    return {"numbers": numbers, "notes": notes, "narrative": narrative}