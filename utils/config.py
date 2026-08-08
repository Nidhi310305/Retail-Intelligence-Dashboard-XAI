from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    app_title: str = "Retail Intelligence Dashboard"
    expected_roles: tuple[str, ...] = (
        "order_date",
        "ship_date",
        "customer_id",
        "sales",
        "profit",
        "discount",
        "quantity",
        "region",
        "category",
    )
