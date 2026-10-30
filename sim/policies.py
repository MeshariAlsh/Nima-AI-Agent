from .optimiser import plan_transfers


def no_redistribution(sim):
    """
    خط الأساس: سلسلة المتاجر كما تعمل اليوم، دون نقل أي شيء.
    Baseline: the chain as it works today. Nobody moves anything.
    """
    return []


def estimate(sim, horizon: int | None = None):
    """
    يحسب الفائض والعجز المتوقعين لكل متجر ومنتج مساء اليوم الحالي.
    Computes expected surplus and need for every store and product tonight.
    """
    sc = sim.sc
    d = sim.day
    h = horizon or sc.delivery_every
    surplus, need = {}, {}

    for s in sc.stores:
        for p in sc.products:
            mu = sc.mean_demand(s.name, p.name)

            # الفائض: نمرّ على الدفعات من الأقدم، ونحسب ما يمكن بيعه قبل انتهاء صلاحية كل دفعة
            # surplus: walk batches oldest first, counting what can sell before each expires
            used = 0.0
            waste = 0.0
            for b in sim._batches(s.name, p.name):
                sellable = max(0.0, mu * (b.expiry_day - d) - used)
                sells = min(b.qty, sellable)
                waste += b.qty - sells
                used += sells
            surplus[(s.name, p.name)] = int(waste)

            # العجز: الطلب المتوقع خلال الأفق ناقص المخزون القابل للبيع ناقص التوريدات المجدولة
            # need: expected demand over the horizon minus usable stock minus scheduled deliveries
            usable = sim.stock(s.name, p.name) - waste
            delivery_qty = round(sc.base_demand[p.name] * sc.delivery_every)
            incoming = sum(delivery_qty for t in range(d + 1, d + h + 1)
                           if t % sc.delivery_every == 0)
            need[(s.name, p.name)] = max(0, int(mu * h - usable - incoming))

    return surplus, need


def solver_only(sim):
    """
    المحسِّن وحده، دون وكيل ذكاء اصطناعي.
    The optimiser on its own, with no AI agent.
    """
    surplus, need = estimate(sim)
    return plan_transfers(sim.sc, surplus, need)