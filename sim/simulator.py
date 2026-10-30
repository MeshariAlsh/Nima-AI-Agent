import numpy as np
import pandas as pd

from .models import Batch, Plan, Scenario, distance_km

FIELDS = ["delivered", "demand", "sold", "sold_discount", "stockout", "wasted",
          "transfer_in", "transfer_out", "discounted", "donated"]
ACTION_COLS = ["day", "type", "src", "dst", "product", "qty", "km", "cost"]


class Simulation:
    def __init__(self, scenario: Scenario, seed: int = 0):
        self.sc = scenario
        # مولّدان عشوائيان منفصلان: الأول لطلب العملاء والثاني لمبيعات التخفيض،
        # حتى يواجه كل نهج العملاء أنفسهم تمامًا
        # two separate random generators: one for customer demand, one for discount sales,
        # so every policy faces exactly the same customers
        self.rng = np.random.default_rng(seed)
        self.rng_clear = np.random.default_rng(seed + 10_000)
        self.batches: list[Batch] = []
        # الوحدات المخفّضة الليلة لتُعرض للبيع غدًا
        # units discounted tonight, on sale tomorrow
        self.clearance: list[Batch] = []
        self.day = 0

    # ---------- دوال للقراءة فقط تستخدمها السياسات ----------
    # ---------- read-only helpers for policies ----------
    def stock(self, store: str, product: str) -> int:
        return sum(b.qty for b in self.batches
                   if b.store == store and b.product == product)

    def expiring_by(self, store: str, product: str, day: int) -> int:
        """
        الوحدات في هذا المتجر التي تنتهي صلاحيتها في اليوم `day` أو قبله.
        Units at this store that expire on or before `day`.
        """
        return sum(b.qty for b in self.batches
                   if b.store == store and b.product == product
                   and b.expiry_day <= day)

    def _batches(self, store: str, product: str) -> list[Batch]:
        """
        دفعات هذا المتجر من المنتج، الأقرب انتهاءً أولًا.
        This store's batches of a product, oldest expiry first.
        """
        return sorted((b for b in self.batches
                       if b.store == store and b.product == product),
                      key=lambda b: b.expiry_day)

    def _take(self, store: str, product: str, qty: int) -> list[tuple[int, int]]:
        """
        يسحب وحدات من أقدم الدفعات ويعيد قائمة (الكمية، يوم الانتهاء).
        Removes units from the oldest batches; returns a list of (qty, expiry_day).
        """
        taken, remaining = [], qty
        for b in self._batches(store, product):
            if remaining == 0:
                break
            t = min(b.qty, remaining)
            b.qty -= t
            remaining -= t
            taken.append((t, b.expiry_day))
        self.batches = [b for b in self.batches if b.qty > 0]
        return taken

    # ---------- المراحل الأربع لليوم ----------
    # ---------- the four stages of a day ----------
    def _deliver(self, rec):
        """
        المستودع يرسل لكل متجر الكمية نفسها، وهي متوسط الطلب.
        Warehouse sends every store the same amount: average demand.
        """
        for s in self.sc.stores:
            for p in self.sc.products:
                qty = round(self.sc.base_demand[p.name] * self.sc.delivery_every)
                self.batches.append(
                    Batch(s.name, p.name, qty, self.day + p.shelf_life - 1))
                rec[(s.name, p.name)]["delivered"] += qty

    def _sell(self, rec):
        """
        أولًا مبيعات التخفيض، ثم طلب عشوائي يُباع فيه المخزون الأقدم أولًا.
        First clearance sales, then random demand where oldest stock sells first.
        """
        # الوحدات المخفّضة: كل وحدة تُباع باحتمال ثابت، وما لا يُباع يُشطب هدرًا
        # discounted units: each sells with a fixed probability; unsold ones are written off
        for c in self.clearance:
            n = int(self.rng_clear.binomial(c.qty, self.sc.discount_sell_prob))
            rec[(c.store, c.product)]["sold_discount"] += n
            rec[(c.store, c.product)]["wasted"] += c.qty - n
        self.clearance = []

        for s in self.sc.stores:
            for p in self.sc.products:
                demand = int(self.rng.poisson(self.sc.mean_demand(s.name, p.name)))
                remaining = demand
                for b in self._batches(s.name, p.name):
                    if remaining == 0:
                        break
                    take = min(b.qty, remaining)
                    b.qty -= take
                    remaining -= take
                r = rec[(s.name, p.name)]
                r["demand"] = demand
                r["sold"] = demand - remaining
                r["stockout"] = remaining
        self.batches = [b for b in self.batches if b.qty > 0]

    def _expire(self, rec):
        """
        كل ما كان اليوم آخر يوم لبيعه يُتلف الليلة.
        Anything whose last sale day is today is thrown away tonight.
        """
        keep = []
        for b in self.batches:
            if b.expiry_day <= self.day:
                rec[(b.store, b.product)]["wasted"] += b.qty
            else:
                keep.append(b)
        self.batches = keep

    def _apply_plan(self, plan: Plan, rec) -> list[dict]:
        """
        ينفّذ خطة الليلة: النقل ثم التخفيض ثم التبرع.
        Executes tonight's plan: transfers, then discounts, then donations.
        """
        actions = []

        # النقل بين الفروع، مع تكلفة رحلة ثابتة لأول شحنة على كل مسار
        # transfers between branches, with a fixed trip cost for the first load on each route
        trips = set()
        for t in plan.transfers:
            if t.src == t.dst or t.qty <= 0:
                continue
            taken = self._take(t.src, t.product, t.qty)
            moved = sum(q for q, _ in taken)
            for q, e in taken:
                self.batches.append(Batch(t.dst, t.product, q, e))
            if moved:
                km = distance_km(self.sc.store(t.src), self.sc.store(t.dst))
                cost = moved * km * self.sc.transport_cost_per_unit_km
                if (t.src, t.dst) not in trips:
                    trips.add((t.src, t.dst))
                    cost += self.sc.trip_cost
                rec[(t.src, t.product)]["transfer_out"] += moved
                rec[(t.dst, t.product)]["transfer_in"] += moved
                actions.append({"day": self.day, "type": "transfer",
                                "src": t.src, "dst": t.dst, "product": t.product,
                                "qty": moved, "km": round(km, 1), "cost": cost})

        # التخفيض: تنتقل الوحدات إلى رف التخفيض لتُعرض غدًا
        # discounts: units move to the clearance shelf for tomorrow
        for c in plan.discounts:
            if c.qty <= 0:
                continue
            taken = self._take(c.store, c.product, c.qty)
            n = sum(q for q, _ in taken)
            for q, e in taken:
                self.clearance.append(Batch(c.store, c.product, q, e))
            if n:
                rec[(c.store, c.product)]["discounted"] += n
                actions.append({"day": self.day, "type": "discount",
                                "src": c.store, "dst": c.store, "product": c.product,
                                "qty": n, "km": 0.0, "cost": 0.0})

        # التبرع: ضمن السعة اليومية لكل جهة مستفيدة، وهي تتولى الاستلام
        # donations: within each recipient's daily capacity; recipients collect
        used = {r.name: 0 for r in self.sc.recipients}
        for g in plan.donations:
            r = self.sc.recipient(g.recipient)
            qty = min(g.qty, r.daily_capacity - used[r.name])
            if qty <= 0:
                continue
            n = sum(q for q, _ in self._take(g.store, g.product, qty))
            if n:
                used[r.name] += n
                rec[(g.store, g.product)]["donated"] += n
                km = distance_km(self.sc.store(g.store), r)
                actions.append({"day": self.day, "type": "donation",
                                "src": g.store, "dst": r.name, "product": g.product,
                                "qty": n, "km": round(km, 1), "cost": 0.0})
        return actions

    # ---------- يوم كامل ----------
    # ---------- one full day ----------
    def step(self, policy):
        rec = {(s.name, p.name): dict.fromkeys(FIELDS, 0)
               for s in self.sc.stores for p in self.sc.products}

        # صباحًا
        # morning
        if self.day % self.sc.delivery_every == 0:
            self._deliver(rec)
        # نهارًا
        # daytime
        self._sell(rec)
        # مساءً
        # evening
        self._expire(rec)
        # مساءً: التخطيط. السياسة القديمة تعيد قائمة نقل فقط
        # evening: plan. Older policies return only a list of transfers
        plan = policy(self)
        if isinstance(plan, list):
            plan = Plan(transfers=plan)
        # ليلًا
        # overnight
        actions = self._apply_plan(plan, rec)

        rows = [{"day": self.day, "store": s, "product": p, **r,
                 "stock_end": self.stock(s, p)}
                for (s, p), r in rec.items()]
        self.day += 1
        return rows, actions, plan.explanation


def simulate(scenario: Scenario, policy, days: int = 30, seed: int = 0):
    """
    تشغيل سياسة واحدة لعدد `days` من الأيام. البذرة نفسها تعني طلب العملاء نفسه.
    Run one policy for `days` days. Same seed = same customer demand.
    """
    sim = Simulation(scenario, seed)
    rows, actions, briefings = [], [], []
    for _ in range(days):
        day = sim.day
        r, a, text = sim.step(policy)
        rows += r
        actions += a
        if text:
            briefings.append({"day": day, "text": text})
    return (pd.DataFrame(rows),
            pd.DataFrame(actions, columns=ACTION_COLS),
            pd.DataFrame(briefings, columns=["day", "text"]))