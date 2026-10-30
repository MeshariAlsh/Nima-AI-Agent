import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Product:
    name: str
    # عدد الأيام من التوريد حتى انتهاء الصلاحية
    # days from delivery until expiry
    shelf_life: int
    # الخسارة بالريال لكل وحدة تُهدر أو عملية بيع تُفقد
    # SAR lost per unit wasted or sale missed
    unit_value: float


@dataclass(frozen=True)
class Store:
    name: str
    lat: float
    lon: float


@dataclass
class Batch:
    store: str
    product: str
    qty: int
    # آخر يوم يمكن فيه بيع هذه الدفعة
    # last day this batch can be sold
    expiry_day: int


@dataclass
class Transfer:
    src: str
    dst: str
    product: str
    qty: int


@dataclass
class Scenario:
    warehouse: Store
    stores: list[Store]
    products: list[Product]
    # المنتج ← متوسط الطلب اليومي في متجر متوسط
    # product -> mean daily units at an average store
    base_demand: dict[str, float]
    # المتجر ← معامل حجم الطلب
    # store -> demand scale
    store_multiplier: dict[str, float]
    # المستودع يورّد كل N أيام
    # warehouse delivers every N days
    delivery_every: int
    transport_cost_per_unit_km: float

    def mean_demand(self, store: str, product: str) -> float:
        return self.base_demand[product] * self.store_multiplier[store]

    def product(self, name: str) -> Product:
        return next(p for p in self.products if p.name == name)

    def store(self, name: str) -> Store:
        return next(s for s in self.stores if s.name == name)


def distance_km(a: Store, b: Store) -> float:
    """
    المسافة المستقيمة بين متجرين (معادلة هافرساين).
    Straight-line distance between two stores (haversine formula).
    """
    r = 6371.0
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a.lat)) * math.cos(math.radians(b.lat))
         * math.sin(dlon / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(h))


def load_scenario(path: str = "data/scenario.json") -> Scenario:
    d = json.loads(Path(path).read_text())
    return Scenario(
        warehouse=Store(**d["warehouse"]),
        stores=[Store(**s) for s in d["stores"]],
        products=[Product(**p) for p in d["products"]],
        base_demand=d["base_demand"],
        store_multiplier=d["store_multiplier"],
        delivery_every=d["delivery_every"],
        transport_cost_per_unit_km=d["transport_cost_per_unit_km"],
    )