import json
from pathlib import Path

from .models import Plan, distance_km

# بيانات حركة المرور المحاكاة. في التشغيل الحقيقي تأتي من واجهة مسارات خرائط Google
# simulated traffic data. In production this comes from the Google Maps Routes API
_traffic = Path("data/traffic.json")
TRAFFIC = json.loads(_traffic.read_text()) if _traffic.exists() else []
# مسافة الطريق أطول من المسافة المستقيمة بنحو 30%، ومتوسط السرعة ليلًا داخل المدينة
# road distance is roughly 30% longer than straight-line, and average night speed in the city
ROAD_FACTOR = 1.3
SPEED_KMH = 35


def route_info(sc, a: str, b: str) -> dict:
    """
    حالة الطريق بين متجرين: المسافة، والزمن، والإغلاقات، والتأخير.
    Road status between two stores: distance, time, closures, delays.
    """
    km = distance_km(sc.store(a), sc.store(b)) * ROAD_FACTOR
    incidents = [t for t in TRAFFIC if [a, b] in t["routes"] or [b, a] in t["routes"]]
    delay = sum(t.get("delay_minutes", 0) for t in incidents)
    return {"road_km": round(km, 1),
            "minutes": round(km / SPEED_KMH * 60 + delay),
            "closed": any(t["type"] == "closure" for t in incidents),
            "incidents": [f"{t['road']}: {t['detail']}" for t in incidents]}


def build_routes(plan: Plan, sc) -> list[dict]:
    """
    يجمع عمليات النقل في شاحنات: شاحنة لكل متجر مصدر، وتُرتَّب نقاط التسليم بالأقرب فالأقرب
    مع تجنّب الطرق المغلقة متى أمكن.
    Groups transfers into trucks: one truck per source store, drops ordered nearest-first,
    avoiding closed roads where possible.
    """
    by_src: dict = {}
    for t in plan.transfers:
        by_src.setdefault(t.src, {}).setdefault(t.dst, []).append((t.product, t.qty))

    routes = []
    for n, (src, drops) in enumerate(sorted(by_src.items()), start=1):
        here, left, stops = src, set(drops), []
        while left:
            # الطرق المفتوحة أولًا، ثم الأقرب
            # open roads first, then the nearest
            nxt = min(left, key=lambda d: (route_info(sc, here, d)["closed"],
                                           route_info(sc, here, d)["road_km"]))
            leg = route_info(sc, here, nxt)
            stops.append({"store": nxt, "from": here, "items": drops[nxt],
                          "units": sum(q for _, q in drops[nxt]),
                          "leg_km": leg["road_km"], "leg_minutes": leg["minutes"],
                          "closed": leg["closed"], "incidents": leg["incidents"]})
            left.remove(nxt)
            here = nxt
        names = [src] + [s["store"] for s in stops]
        routes.append({
            "truck": n, "start": src, "stops": stops, "names": names,
            "path": [[sc.store(x).lon, sc.store(x).lat] for x in names],
            "units": sum(s["units"] for s in stops),
            "km": round(sum(s["leg_km"] for s in stops), 1),
            "minutes": sum(s["leg_minutes"] for s in stops),
            "closed_legs": [f"{s['from']} -> {s['store']}" for s in stops if s["closed"]],
        })
    return routes