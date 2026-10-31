import pulp

from .models import Discount, Donation, Plan, Scenario, Transfer, distance_km

# الأولويات: كم ريالًا يساوي تجنّب هدر وحدة واحدة، ومنع نفاد وحدة واحدة
# priorities: how many SAR avoiding one wasted unit, and preventing one stockout, are worth
PRIORITIES = {
    # استرداد القيمة المالية فقط
    # recover financial value only
    "value":    {"waste": 0.0,  "availability": 0.0},
    # موازنة بين القيمة والهدر والتوفر
    # balance value, waste and availability
    "balanced": {"waste": 3.0,  "availability": 3.0},
    # تقليل الهدر أولًا
    # reduce waste first
    "waste":    {"waste": 15.0, "availability": 1.0},
}


def plan_actions(scenario: Scenario, surplus: dict, need: dict,
                 priority: str = "balanced",
                 urgent: dict | None = None,
                 allow_transfer: bool = True,
                 allow_discount: bool = True,
                 allow_donation: bool = True,
                 no_discount: set | None = None,
                 max_trips: int | None = None,
                 budget: float | None = None,
                 blocked_routes: set | None = None,
                 blocked_recipients: set | None = None) -> Plan:
    """
    يختار لكل فائض أفضل توزيع بين النقل والتخفيض والتبرع، وفق أولويات صاحب القرار وقيود العمليات.
    For every surplus, chooses the best split between transfer, discount and donation,
    following the decision-maker's priorities and the operations constraints.
    """
    w = PRIORITIES[priority]
    no_discount = no_discount or set()
    blocked_routes = blocked_routes or set()
    blocked_recipients = blocked_recipients or set()
    stores = [s.name for s in scenario.stores]
    products = [p.name for p in scenario.products]
    value = {p.name: p.unit_value for p in scenario.products}
    q = scenario.discount_sell_prob
    keep = 1 - scenario.discount_pct

    # المسارات المسموحة فقط، بعد استبعاد الطرق المغلقة
    # allowed routes only, after removing closed roads
    routes = [(i, j) for i in stores for j in stores
              if i != j and (i, j) not in blocked_routes]

    # تكلفة نقل الوحدة الواحدة على كل مسار بالريال
    # cost of moving one unit along each route, in SAR
    unit_cost = {(i, j): distance_km(scenario.store(i), scenario.store(j))
                 * scenario.transport_cost_per_unit_km for (i, j) in routes}

    # الفائض العاجل (ما ينتهي غدًا) هو وحده المؤهل للتخفيض أو التبرع
    # urgent surplus (expiring tomorrow) is the only surplus eligible for discount or donation
    if urgent is None:
        urgent = surplus
    has_surplus = [(i, p) for i in stores for p in products
                   if surplus.get((i, p), 0) > 0]
    if not has_surplus:
        return Plan()
    has_urgent = [(i, p) for (i, p) in has_surplus if urgent.get((i, p), 0) > 0]
    recipients = [r for r in scenario.recipients if r.name not in blocked_recipients]

    # متغيرات القرار: النقل (x)، والتخفيض (c)، والتبرع (g)
    # decision variables: transfer (x), discount (c), donation (g)
    x = ({(i, j, p): pulp.LpVariable(f"x_{i}_{j}_{p}", 0, None, pulp.LpContinuous)
          for (i, j) in routes for p in products
          if surplus.get((i, p), 0) > 0 and need.get((j, p), 0) > 0}
         if allow_transfer else {})
    c = ({(i, p): pulp.LpVariable(f"c_{i}_{p}", 0, None, pulp.LpContinuous)
          for (i, p) in has_urgent if (i, p) not in no_discount}
         if allow_discount else {})
    g = ({(i, r.name, p): pulp.LpVariable(f"g_{i}_{k}_{p}", 0, None, pulp.LpContinuous)
          for (i, p) in has_urgent for k, r in enumerate(recipients)}
         if allow_donation else {})
    if not (x or c or g):
        return Plan()

     # متغير ثنائي لكل متجر مصدر: هل تخرج منه شاحنة الليلة؟ الشاحنة الواحدة قد تمر بعدة متاجر
    # one binary per source store: does a truck leave it tonight? One truck can drop at several stores
    y = {i: pulp.LpVariable(f"y_{i}", 0, 1, pulp.LpBinary)
         for i in {i for (i, _, _) in x}}

    prob = pulp.LpProblem("nima_actions", pulp.LpMaximize)

    # الهدف بالريال:
    # النقل: قيمة البيع بالسعر الكامل + قيمة تجنّب الهدر + قيمة منع النفاد − تكلفة النقل
    # التخفيض: الإيراد المتوقع بعد الخصم + قيمة الهدر المتجنَّب المتوقع
    # التبرع: قيمة تجنّب الهدر
    # objective, in SAR:
    # transfer: full-price sale + waste avoided + stockout prevented − transport
    # discount: expected discounted revenue + expected waste avoided
    # donation: waste avoided
    prob += (
        pulp.lpSum((value[p] - unit_cost[(i, j)] + w["waste"] + w["availability"]) * v
                   for (i, j, p), v in x.items())
        - pulp.lpSum(scenario.trip_cost * v for v in y.values())
        + pulp.lpSum((q * keep * value[p] + q * w["waste"]) * v
                     for (i, p), v in c.items())
        + pulp.lpSum(w["waste"] * v for v in g.values())
    )

    # قيد: مجموع ما يُنقل ويُخفَّض ويُتبرَّع به لا يتجاوز الفائض
    # constraint: transferred + discounted + donated can't exceed the surplus
    for (i, p) in has_surplus:
        parts = [v for (a, _, b), v in x.items() if a == i and b == p]
        if (i, p) in c:
            parts.append(c[(i, p)])
        parts += [v for (a, _, b), v in g.items() if a == i and b == p]
        if parts:
            prob += pulp.lpSum(parts) <= surplus[(i, p)]

    # قيد: التخفيض والتبرع معًا لا يتجاوزان الفائض العاجل
    # constraint: discount plus donation can't exceed the urgent surplus
    for (i, p) in has_urgent:
        parts = ([c[(i, p)]] if (i, p) in c else []) + \
                [v for (a, _, b), v in g.items() if a == i and b == p]
        if parts:
            prob += pulp.lpSum(parts) <= urgent[(i, p)]

    # قيد: لا يستقبل المتجر أكثر من حاجته
    # constraint: a store can't receive more than it needs
    for (j, p), qty in need.items():
        inc = [v for (_, b, r), v in x.items() if b == j and r == p]
        if inc:
            prob += pulp.lpSum(inc) <= qty

    # قيد: لا تتجاوز التبرعات السعة اليومية لكل جهة مستفيدة متاحة
    # constraint: donations can't exceed each available recipient's daily capacity
    for r in recipients:
        parts = [v for (_, name, _), v in g.items() if name == r.name]
        if parts:
            prob += pulp.lpSum(parts) <= r.daily_capacity

       # قيد: لا يُحمَّل شيء إلا إذا خرجت شاحنة، وبحد أقصى سعة الشاحنة
    # constraint: nothing is loaded unless a truck leaves, and at most one truck's capacity
    for i in y:
        prob += (pulp.lpSum(v for (a, _, _), v in x.items() if a == i)
                 <= scenario.truck_capacity * y[i])

    # قيد العمليات: عدد الشاحنات المتاحة الليلة
    # operations constraint: trucks available tonight
    if max_trips is not None and y:
        prob += pulp.lpSum(y.values()) <= max_trips

    # قيد العمليات: ميزانية النقل الليلة
    # operations constraint: tonight's transport budget
    if budget is not None and x:
        prob += (pulp.lpSum(unit_cost[(i, j)] * v for (i, j, _), v in x.items())
                 + scenario.trip_cost * pulp.lpSum(y.values())) <= budget

    # حد زمني لكل مسألة، وقبول حل ضمن 1% من الأمثل
    # time limit per problem, and accept a solution within 1% of optimal
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5, gapRel=0.01))

    def units(v):
        # التقريب إلى الأسفل يضمن عدم تجاوز الفائض أو الحاجة
        # rounding down guarantees we never exceed the surplus or the need
        return int(v.value() or 0)

    return Plan(
        transfers=[Transfer(i, j, p, units(v)) for (i, j, p), v in x.items() if units(v) > 0],
        discounts=[Discount(i, p, units(v)) for (i, p), v in c.items() if units(v) > 0],
        donations=[Donation(i, p, units(v), r) for (i, r, p), v in g.items() if units(v) > 0],
    )