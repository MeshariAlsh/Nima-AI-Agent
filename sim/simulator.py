import numpy as np
import pandas as pd

from .models import Batch, Scenario, Transfer, distance_km

FIELDS = ["delivered", "demand", "sold", "stockout", "wasted",
          "transfer_in", "transfer_out"]


class Simulation:
    def __init__(self, scenario: Scenario, seed: int = 0):
        self.sc = scenario
        self.rng = np.random.default_rng(seed)
        self.batches: list[Batch] = []
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
        طلب عشوائي، ويُباع المخزون الأقدم أولًا.
        Random demand; oldest stock sells first.
        """
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

    def _apply_transfers(self, transfers: list[Transfer], rec) -> list[dict]:
        """
        نقل المخزون ليلًا. تُنقل الدفعات الأقدم أولًا مع الاحتفاظ بتاريخ صلاحيتها.
        Move stock overnight. Oldest batches move first, keeping their expiry.
        """
        moves = []
        for t in transfers:
            if t.src == t.dst or t.qty <= 0:
                continue
            remaining = t.qty
            for b in self._batches(t.src, t.product):
                if remaining == 0:
                    break
                take = min(b.qty, remaining)
                b.qty -= take
                remaining -= take
                self.batches.append(Batch(t.dst, t.product, take, b.expiry_day))
            moved = t.qty - remaining
            if moved:
                km = distance_km(self.sc.store(t.src), self.sc.store(t.dst))
                cost = moved * km * self.sc.transport_cost_per_unit_km
                rec[(t.src, t.product)]["transfer_out"] += moved
                rec[(t.dst, t.product)]["transfer_in"] += moved
                moves.append({"day": self.day, "src": t.src, "dst": t.dst,
                              "product": t.product, "qty": moved,
                              "km": round(km, 1), "cost": cost})
        self.batches = [b for b in self.batches if b.qty > 0]
        return moves

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
        # مساءً: التخطيط
        # evening: plan
        transfers = policy(self)
        # ليلًا
        # overnight
        moves = self._apply_transfers(transfers, rec)

        rows = [{"day": self.day, "store": s, "product": p, **r,
                 "stock_end": self.stock(s, p)}
                for (s, p), r in rec.items()]
        self.day += 1
        return rows, moves


def simulate(scenario: Scenario, policy, days: int = 30, seed: int = 0):
    """
    تشغيل سياسة واحدة لعدد `days` من الأيام. البذرة نفسها تعني طلب العملاء نفسه.
    Run one policy for `days` days. Same seed = same customer demand.
    """
    sim = Simulation(scenario, seed)
    rows, moves = [], []
    for _ in range(days):
        r, m = sim.step(policy)
        rows += r
        moves += m
    log = pd.DataFrame(rows)
    moves = pd.DataFrame(moves, columns=["day", "src", "dst", "product",
                                         "qty", "km", "cost"])
    return log, moves