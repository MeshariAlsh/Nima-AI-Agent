from .optimiser import plan_actions


def no_redistribution(sim):
    """
    خط الأساس: سلسلة المتاجر كما تعمل اليوم، دون نقل أي شيء.
    Baseline: the chain as it works today. Nobody moves anything.
    """
    return []


def estimate(sim, horizon: int | None = None, demand_adj: dict | None = None,
             no_delivery: set | None = None, spoil: set | None = None):
    """
    يحسب الفائض والفائض العاجل والعجز المتوقع لكل متجر ومنتج مساء اليوم الحالي.
    يستطيع الوكيل تعديل التوقعات: معامل الطلب، والتوريدات الملغاة، والمخزون المعرّض للتلف.
    Computes surplus, urgent surplus and expected need for every store and product tonight.
    The agent can adjust the inputs: demand multipliers, cancelled deliveries, stock at risk of spoiling.
    """
    sc = sim.sc
    d = sim.day
    h = horizon or sc.delivery_every
    demand_adj = demand_adj or {}
    no_delivery = no_delivery or set()
    spoil = spoil or set()
    surplus, urgent, need = {}, {}, {}

    for s in sc.stores:
        for p in sc.products:
            key = (s.name, p.name)
            mu = sc.mean_demand(s.name, p.name) * demand_adj.get(key, 1.0)

            # مخزون سيتلف الليلة: كله فائض يجب أن يغادر المتجر، والمتجر لا يحتاج المزيد
            # stock that will spoil tonight: all of it is surplus that must leave, and the store needs none
            if key in spoil:
                total = sim.stock(s.name, p.name)
                surplus[key], urgent[key], need[key] = total, total, 0
                continue

            # الفائض: نمرّ على الدفعات من الأقدم، ونحسب ما يمكن بيعه قبل انتهاء صلاحية كل دفعة.
            # الفائض العاجل هو ما ينتهي غدًا، وهو وحده المؤهل للتخفيض أو التبرع
            # surplus: walk batches oldest first, counting what can sell before each expires.
            # urgent surplus expires tomorrow, and is the only surplus eligible for discount or donation
            used = 0.0
            waste = 0.0
            waste_urgent = 0.0
            for b in sim._batches(s.name, p.name):
                sellable = max(0.0, mu * (b.expiry_day - d) - used)
                sells = min(b.qty, sellable)
                waste += b.qty - sells
                if b.expiry_day <= d + 1:
                    waste_urgent += b.qty - sells
                used += sells
            surplus[key] = int(waste)
            urgent[key] = int(waste_urgent)

            # العجز: الطلب المتوقع خلال الأفق ناقص المخزون القابل للبيع ناقص التوريدات المجدولة
            # need: expected demand over the horizon minus usable stock minus scheduled deliveries
            usable = sim.stock(s.name, p.name) - waste
            delivery_qty = round(sc.base_demand[p.name] * sc.delivery_every)
            incoming = sum(delivery_qty for t in range(d + 1, d + h + 1)
                           if t % sc.delivery_every == 0 and (p.name, t) not in no_delivery)
            need[key] = max(0, int(mu * h - usable - incoming))

    return surplus, urgent, need


def make_solver_policy(priority: str = "balanced", transfer: bool = True,
                       discount: bool = True, donation: bool = True):
    """
    ينشئ سياسة تعتمد على المحسِّن وحده بأولوية محددة وإجراءات مختارة. لا تقرأ الملاحظات.
    Builds an optimiser-only policy with a given priority and chosen actions. It can't read notes.
    """
    def policy(sim):
        surplus, urgent, need = estimate(sim)
        return plan_actions(sim.sc, surplus, need, priority, urgent=urgent,
                            allow_transfer=transfer, allow_discount=discount,
                            allow_donation=donation)
    return policy