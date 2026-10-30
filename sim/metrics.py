from .models import Scenario


def summarise(sc: Scenario, log, actions) -> dict:
    """
    يحوّل سجلّ المحاكاة إلى مؤشرات رئيسية، بالوحدات وبالريال.
    Turns a simulation log into headline metrics, in units and in SAR.
    """
    price = {p.name: p.unit_value for p in sc.products}
    keep = 1 - sc.discount_pct
    pr = log["product"].map(price)

    # الإيراد: مبيعات بالسعر الكامل + مبيعات التخفيض بعد الخصم
    # revenue: full-price sales + clearance sales after the discount
    revenue_full = float((log["sold"] * pr).sum())
    revenue_clear = float((log["sold_discount"] * pr * keep).sum())
    transport = float(actions["cost"].sum())

    return {
        "wasted": int(log["wasted"].sum()),
        "stockout": int(log["stockout"].sum()),
        "donated": int(log["donated"].sum()),
        # قيمة الطعام المُهدَر بسعر البيع
        # value of the wasted food at selling price
        "waste_value": float((log["wasted"] * pr).sum()),
        # قيمة المبيعات الضائعة بسبب نفاد المخزون
        # value of sales lost to stockouts
        "lost_sales_value": float((log["stockout"] * pr).sum()),
        "revenue": revenue_full + revenue_clear,
        "transport": transport,
        # صافي الإيراد بعد النقل. التوريدات متطابقة في كل السياسات،
        # لذا فإن فرق صافي الإيراد يساوي فرق الربح
        # net revenue after transport. Deliveries are identical across policies,
        # so the difference in net revenue equals the difference in profit
        "net_revenue": revenue_full + revenue_clear - transport,
    }