import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .models import Plan
from .optimiser import plan_actions
from .policies import estimate

load_dotenv()

# النموذج ومفتاح الواجهة من ملف .env
# model and API key come from the .env file
MODEL = os.getenv("NIMA_MODEL", "gpt-4.1-mini")
BASE_URL = os.getenv("NIMA_BASE_URL") or None
API_KEY = os.getenv("OPENAI_API_KEY", "")
REASONING_EFFORT = os.getenv("NIMA_REASONING_EFFORT", "none")

# أقصى عدد من جولات التفكير والأدوات في الليلة الواحدة
# maximum rounds of reasoning and tool calls per night
MAX_STEPS = 8

SYSTEM_PROMPT = """You are Ni'ma, the operations agent for a grocery chain in Riyadh.
Stores: {stores}. Products: {products}.

Every evening an optimiser plans what to do with food at risk of expiring: transfer it to
another branch, discount it locally, or donate it to a food bank. The optimiser does all
the arithmetic. Your job is judgement: read tonight's manager notes (Arabic or English),
turn them into changes to the optimiser's inputs with your tools, run the plan, check it,
and explain it.

Tonight is day {day}. "Tomorrow" means day {tomorrow}.

Tools:
- get_overview: tonight's surplus and shortages per store.
- adjust_demand: expected demand at a store will change on specific days
  (multiplier between 0.3 and 3.0). Use it for events that raise or lower sales.
- close_store: a store will make no sales on specific days.
- cancel_delivery: a warehouse delivery of a product will not arrive on specific days.
- mark_spoilage: stock of products at a store will be lost tonight, so it must leave.
- run_plan: run the optimiser with your adjustments.
- finish: submit the plan with your briefing.

Rules:
- Adjust only for what the notes actually say. If a note is vague, make the smallest
  reasonable adjustment and say so in the briefing.
- Never state quantities you did not get from run_plan.
- Always call run_plan before finish.
- Briefing: 3 to 5 short sentences in English for the operations manager: what the notes
  said, what you changed, what the plan does, and anything needing human attention.
  Then one line in Arabic summarising tonight's plan for store managers."""


def tool_schemas(sc) -> list[dict]:
    """
    تعريفات الأدوات التي يراها النموذج اللغوي.
    The tool definitions the language model sees.
    """
    stores = [s.name for s in sc.stores]
    products = [p.name for p in sc.products] + ["all"]
    store = {"type": "string", "enum": stores}
    prods = {"type": "array", "items": {"type": "string", "enum": products}}
    days = {"type": "array", "items": {"type": "integer"}}

    def fn(name, description, props, required):
        return {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}}

    return [
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
        fn("run_plan", "Run the optimiser with the current adjustments and see the plan.",
           {"priority": {"type": "string", "enum": ["value", "balanced", "waste"]}}, []),
        fn("finish", "Submit tonight's plan with your briefing.",
           {"briefing": {"type": "string"}}, ["briefing"]),
    ]


def summarise_plan(plan: Plan) -> dict:
    """
    ملخص قصير للخطة يعود إلى النموذج، دون إغراقه بالتفاصيل.
    A short plan summary returned to the model, without flooding it with detail.
    """
    return {
        "transferred_units": sum(t.qty for t in plan.transfers),
        "discounted_units": sum(c.qty for c in plan.discounts),
        "donated_units": sum(g.qty for g in plan.donations),
        "transfers": [f"{t.qty} {t.product} {t.src} -> {t.dst}" for t in plan.transfers][:15],
        "discounts": [f"{c.qty} {c.product} at {c.store}" for c in plan.discounts][:10],
        "donations": [f"{g.qty} {g.product} {g.store} -> {g.recipient}"
                      for g in plan.donations][:10],
    }


def routine_briefing(plan: Plan) -> str:
    s = summarise_plan(plan)
    return (f"Routine night, no notes. Moved {s['transferred_units']} units between "
            f"branches, discounted {s['discounted_units']}, donated {s['donated_units']}.")


class NimaAgent:
    """
    وكيل نعمة: يقرأ الملاحظات، ويعدّل مدخلات المحسِّن، ويشغّله، ويشرح الخطة.
    Ni'ma agent: reads notes, adjusts the optimiser's inputs, runs it, and explains the plan.
    """

    def __init__(self, priority: str = "balanced", use_llm: bool = True):
        self.priority = priority
        self.use_llm = use_llm
        # تعديلات الطلب تبقى سارية للأيام المحددة
        # demand adjustments stay in force for their days
        self.demand_events: list[dict] = []
        # التوريدات الملغاة: (المنتج، اليوم)
        # cancelled deliveries: (product, day)
        self.no_delivery: set = set()
        # المخزون المعرّض للتلف الليلة فقط: (المتجر، المنتج)
        # stock at risk of spoiling tonight only: (store, product)
        self.spoil: set = set()
        # ملاحظات تُضاف مباشرة من التطبيق في العرض الحي
        # notes injected straight from the app in the live demo
        self.extra_notes: list[dict] = []
        self.transcript: list[dict] = []
        self.last_plan: Plan | None = None
        self._client = None

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
                            urgent=urgent, no_discount=self.spoil)

    def _overview(self, sim) -> dict:
        surplus, _, need = self._inputs(sim)
        return {s.name: {
            "surplus": {p: v for (st, p), v in surplus.items() if st == s.name and v > 0},
            "need": {p: v for (st, p), v in need.items() if st == s.name and v > 0},
        } for s in sim.sc.stores}

    @staticmethod
    def _expand(products: list, sc) -> list[str]:
        names = [p.name for p in sc.products]
        return names if "all" in products else [p for p in products if p in names]

    # ---------- تنفيذ الأدوات ----------
    # ---------- tool execution ----------
    def _call_tool(self, sim, name: str, args: dict, state: dict) -> dict:
        sc = sim.sc
        stores = [s.name for s in sc.stores]
        if "store" in args and args["store"] not in stores:
            return {"error": f"unknown store {args['store']}"}
        days = [int(x) for x in args.get("days", [])]

        if name == "get_overview":
            return self._overview(sim)
        if name == "adjust_demand":
            # حدود آمنة للمعامل، حتى لا يبالغ النموذج في التوقعات
            # safe limits on the multiplier, so the model can't overreact
            m = min(3.0, max(0.3, float(args.get("multiplier", 1.0))))
            self.demand_events.append({"store": args["store"], "multiplier": m, "days": days,
                                       "products": self._expand(args.get("products", []), sc)})
            return {"ok": True, "applied_multiplier": m}
        if name == "close_store":
            self.demand_events.append({"store": args["store"], "multiplier": 0.0,
                                       "days": days, "products": self._expand(["all"], sc)})
            return {"ok": True}
        if name == "cancel_delivery":
            for p in self._expand(args.get("products", []), sc):
                for t in days:
                    self.no_delivery.add((p, t))
            return {"ok": True}
        if name == "mark_spoilage":
            for p in self._expand(args.get("products", []), sc):
                self.spoil.add((args["store"], p))
            return {"ok": True}
        if name == "run_plan":
            state["plan"] = self._plan(sim, args.get("priority"))
            return summarise_plan(state["plan"])
        if name == "finish":
            state["briefing"] = args.get("briefing", "")
            return {"ok": True}
        return {"error": f"unknown tool {name}"}

    # ---------- حلقة الوكيل ----------
    # ---------- the agent loop ----------
    def _run_llm(self, sim, notes: list[dict]) -> Plan:
        sc = sim.sc
        system = SYSTEM_PROMPT.format(
            stores=", ".join(s.name for s in sc.stores),
            products=", ".join(p.name for p in sc.products),
            day=sim.day, tomorrow=sim.day + 1)
        user = "Tonight's notes:\n" + "\n".join(f"- [{n['store']}] {n['text']}" for n in notes)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.transcript.append({"day": sim.day, "role": "notes", "content": user})
        state = {"plan": None, "briefing": None}
        reply = ""

        for _ in range(MAX_STEPS):
            extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
            resp = self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=tool_schemas(sc),
                tool_choice="auto", **extra)
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
            if state["briefing"] is not None:
                break

        # إذا لم يشغّل الوكيل المحسِّن، نشغّله بتعديلاته
        # if the agent never ran the optimiser, run it with its adjustments
        plan = state["plan"] or self._plan(sim)
        plan.explanation = state["briefing"] or reply or "Plan created by the optimiser."
        return plan

    def __call__(self, sim) -> Plan:
        """
        يُستدعى كل مساء. النموذج اللغوي يعمل فقط في الليالي التي فيها ملاحظات.
        Called every evening. The language model runs only on nights with notes.
        """
        self.spoil = set()
        notes = sim.notes_today() + self.extra_notes
        self.extra_notes = []
        if notes and self.use_llm:
            try:
                plan = self._run_llm(sim, notes)
            except Exception as e:
                # عند تعطل النموذج: نستخدم خطة المحسِّن ونبلّغ بذلك
                # if the model fails: use the optimiser's plan and say so
                plan = self._plan(sim)
                plan.explanation = (f"AI agent unavailable tonight ({type(e).__name__}: "
                                    f"{str(e)[:300]}). The optimiser's plan was used "
                                    "without note adjustments.")
        else:
            plan = self._plan(sim)
            plan.explanation = routine_briefing(plan)
        self.last_plan = plan
        return plan


def live_plan(sc, note_store: str, note_text: str, day: int,
              priority: str = "balanced", seed: int = 0):
    """
    للعرض الحي: يشغّل المحاكاة حتى اليوم المحدد، ثم يطلب من الوكيل التخطيط حول ملاحظة جديدة.
    For the live demo: runs the simulation up to a day, then asks the agent to plan around a new note.
    """
    from .policies import make_solver_policy
    from .simulator import Simulation

    sim = Simulation(sc, seed)
    warmup = make_solver_policy(priority)
    for _ in range(day):
        sim.step(warmup)
    agent = NimaAgent(priority)
    agent.extra_notes = [{"store": note_store, "text": note_text}]
    rows, actions, briefing = sim.step(agent)
    return rows, actions, briefing, agent.transcript