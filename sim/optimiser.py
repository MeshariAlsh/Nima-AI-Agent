import pulp

from .models import Scenario, Transfer, distance_km


def plan_transfers(scenario: Scenario, surplus: dict, need: dict,
                   trip_cost: float = 20.0) -> list[Transfer]:
    """
    يخطط لعمليات النقل الأعلى قيمة بين المتاجر باستخدام البرمجة الخطية بالأعداد الصحيحة.
    Plans the most valuable transfers between stores using integer linear programming.

    surplus[(store, product)]: وحدات ستنتهي صلاحيتها دون بيع
    surplus[(store, product)]: units that will expire unsold
    need[(store, product)]: العجز المتوقع
    need[(store, product)]: expected shortfall
    """
    stores = [s.name for s in scenario.stores]
    value = {p.name: p.unit_value for p in scenario.products}
    routes = [(i, j) for i in stores for j in stores if i != j]

    # تكلفة نقل الوحدة الواحدة على كل مسار بالريال
    # cost of moving one unit along each route, in SAR
    unit_cost = {(i, j): distance_km(scenario.store(i), scenario.store(j))
                 * scenario.transport_cost_per_unit_km for (i, j) in routes}

    # متغيرات القرار: عدد الوحدات المنقولة لكل مسار ومنتج،
    # فقط حيث يوجد فائض عند المصدر وعجز عند الوجهة
    # decision variables: units moved per route and product,
    # only where the source has surplus and the destination has need
    x = {(i, j, p): pulp.LpVariable(f"x_{i}_{j}_{p}", 0, None, pulp.LpInteger)
         for (i, j) in routes for p in value
         if surplus.get((i, p), 0) > 0 and need.get((j, p), 0) > 0}
    if not x:
        return []

    # متغير ثنائي لكل مسار: هل تخرج شاحنة على هذا المسار الليلة؟
    # one binary variable per route: does a truck run on this route tonight?
    used_routes = {(i, j) for (i, j, _) in x}
    y = {r: pulp.LpVariable(f"y_{r[0]}_{r[1]}", 0, 1, pulp.LpBinary) for r in used_routes}

    prob = pulp.LpProblem("nima_transfers", pulp.LpMaximize)

    # الهدف: قيمة الطعام المُنقَذ ناقص تكلفة نقل الوحدات ناقص تكلفة الرحلات
    # objective: value of food saved minus per-unit transport minus truck trips
    prob += (pulp.lpSum((value[p] - unit_cost[(i, j)]) * v
                        for (i, j, p), v in x.items())
             - pulp.lpSum(trip_cost * y[r] for r in y))

    # قيد: لا يرسل المتجر أكثر من فائضه
    # constraint: a store can't send more than its surplus
    for (i, p), qty in surplus.items():
        out = [v for (a, _, q), v in x.items() if a == i and q == p]
        if out:
            prob += pulp.lpSum(out) <= qty

    # قيد: لا يستقبل المتجر أكثر من حاجته
    # constraint: a store can't receive more than it needs
    for (j, p), qty in need.items():
        inc = [v for (_, b, q), v in x.items() if b == j and q == p]
        if inc:
            prob += pulp.lpSum(inc) <= qty

    # قيد: لا يُنقل شيء على مسار إلا إذا خرجت شاحنة عليه
    # constraint: nothing moves on a route unless a truck runs on it
    big_m = sum(surplus.values())
    for r in y:
        prob += pulp.lpSum(v for (i, j, _), v in x.items() if (i, j) == r) <= big_m * y[r]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return [Transfer(i, j, p, int(round(v.value())))
            for (i, j, p), v in x.items()
            if v.value() is not None and v.value() > 0.5]