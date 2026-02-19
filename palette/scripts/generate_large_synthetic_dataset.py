import argparse
import json
from pathlib import Path


def read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def pick(values: list[str], seed: int) -> str:
    return values[seed % len(values)]


def sustainability_pct(grade: str, seed: int) -> str:
    buckets = {"A": (88, 98), "B": (74, 89), "C": (60, 78)}
    low, high = buckets.get(grade, (70, 85))
    value = low + (seed % (high - low + 1))
    return f"{value}%"


def role_candidates(family: str) -> list[str]:
    by_family = {
        "Upper": ["Upper Base", "Upper Overlay", "Upper Reinforcement"],
        "Lining": ["Lining", "Inner Lining"],
        "Midsole": ["Primary Midsole", "Midsole Insert"],
        "Midsole Plate": ["Plate", "Propulsion Plate"],
        "Outsole": ["Outsole Forefoot", "Outsole Heel", "Outsole"],
        "Insole": ["Insole", "Heel Pad Insole"],
        "Reinforcement": ["Toe Protection", "Heel Counter", "Midfoot Support"],
        "Lace": ["Lace", "Lace System"],
    }
    return by_family.get(family, [family or "Material"])


def unit_for_family(family: str) -> str:
    by_family = {
        "Upper": "meter",
        "Lining": "meter",
        "Midsole": "kg",
        "Midsole Plate": "piece",
        "Outsole": "kg",
        "Insole": "pair",
        "Reinforcement": "piece",
        "Lace": "pair",
    }
    return by_family.get(family, "unit")


def build_dataset(root: Path, target_size: int) -> list[dict]:
    materials = read_json(root / "materials.json")
    products = read_json(root / "products.json")
    links = read_json(root / "product_materials.json")

    products_by_id = {safe_int(p["id"]): p for p in products}
    materials_by_id = {safe_int(m["id"]): m for m in materials}

    suppliers = [
        "AeroComposite",
        "TextilePro Inc.",
        "CushionTech Ltd.",
        "GreenWeave Labs",
        "PrimeFoam Systems",
        "Velocity Materials",
        "EcoPolymers",
    ]
    strength_by_focus = {
        "Breathability": "Flexible",
        "Lightweight": "Flexible",
        "Race-Day Lightweight": "Rigid",
        "Adaptive Fit": "Flexible",
        "Support": "Stable",
        "Premium Look": "Stable",
        "Casual Style": "Comfort",
        "Sustainable Knit": "Flexible",
        "Breathable Support": "Stable",
        "Lifestyle Premium": "Stable",
        "Moisture Management": "Comfort",
        "Comfort": "Comfort",
        "Warmth": "Comfort",
        "Cooling": "Flexible",
        "Natural Feel": "Comfort",
        "Cushioning": "Stable",
        "Sustainable Cushioning": "Stable",
        "Energy Return": "Durable",
        "Propulsion": "Rigid",
        "Stability": "Stable",
        "Abrasion Resistance": "Durable",
        "Off-Road Traction": "Durable",
        "Court Grip": "Durable",
        "Impact Protection": "Durable",
        "Trail Protection": "Durable",
    }
    durability_by_family = {
        "Upper": "Medium",
        "Lining": "Medium",
        "Midsole": "High",
        "Midsole Plate": "High",
        "Outsole": "High",
        "Insole": "Medium",
        "Reinforcement": "High",
        "Lace": "Medium",
    }

    generated = []
    seen_ids = set()

    def add_record(product: dict, material: dict, role: str, score_float: float, variant: str) -> None:
        product_id = safe_int(product.get("id"))
        material_id = safe_int(material.get("id"))
        seed = (product_id * 997) + (material_id * 389) + sum(ord(c) for c in variant)
        focus = material.get("performanceFocus", "Balanced")
        family = material.get("family", "Material")
        base_cost = safe_float(material.get("cost"), 1.0)
        adjusted_cost = base_cost * (0.92 + ((seed % 19) * 0.01))
        score = int(round(max(0.60, min(0.99, score_float)) * 100))

        rec_id = f"p{product_id}_m{material_id}_{variant}".replace(" ", "_").lower()
        if rec_id in seen_ids:
            rec_id = f"{rec_id}_{seed % 97}"
        seen_ids.add(rec_id)

        lower_focus = str(focus).lower()
        if "light" in lower_focus or "race" in lower_focus:
            weight = "Lightweight"
        elif "protection" in lower_focus or "stability" in lower_focus:
            weight = "Medium"
        else:
            weight = pick(["Lightweight", "Medium"], seed)

        generated.append(
            {
                "id": rec_id,
                "name": material.get("name", "Synthetic Material"),
                "type": role,
                "category": product.get("category", "General"),
                "department": product.get("department", "Running"),
                "division": product.get("division", "Performance"),
                "supplier": pick(suppliers, seed),
                "sustainability": sustainability_pct(material.get("sustainabilityGrade", "B"), seed),
                "cost": f"${adjusted_cost:.2f} / {unit_for_family(family)}",
                "score": max(60, min(99, score)),
                "strength": strength_by_focus.get(focus, pick(["Stable", "Flexible", "Durable"], seed)),
                "weight": weight,
                "durability": durability_by_family.get(family, "Medium"),
                "description": (
                    f"{focus} setup for {product.get('name', 'footwear model')} "
                    f"({product.get('season', 'SP26')}) tuned for {role.lower()} use."
                ),
            }
        )

    for link in links:
        product = products_by_id.get(safe_int(link.get("productId")))
        material = materials_by_id.get(safe_int(link.get("materialId")))
        if not product or not material:
            continue
        role = str(link.get("role") or material.get("family") or "Material")
        add_record(product, material, role, safe_float(link.get("score"), 0.80), "linked")

    idx = 0
    product_count = len(products)
    material_count = len(materials)
    max_iterations = max(target_size * 6, 1000)
    while len(generated) < target_size and idx < max_iterations:
        product = products[(idx * 5 + (idx // 11)) % product_count]
        material = materials[(idx * 13 + (idx // 7)) % material_count]
        roles = role_candidates(str(material.get("family", "Material")))
        role = roles[(idx + safe_int(product.get("id"))) % len(roles)]

        score = 0.67 + (((idx * 19) + safe_int(product.get("id")) + safe_int(material.get("id"))) % 33) / 100.0
        if product.get("division") == "Performance":
            score += 0.05
        elif product.get("division") in {"Outdoor", "Lifestyle"}:
            score += 0.03
        add_record(product, material, role, min(score, 0.99), f"syn{idx:05d}")

        if len(generated) < target_size and idx % 3 == 0:
            alt_role = roles[(idx + 1) % len(roles)]
            add_record(product, material, alt_role, max(0.60, score - 0.07), f"alt{idx:05d}")

        idx += 1

    return sorted(generated[:target_size], key=lambda x: x["score"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Generate large synthetic material recommendation dataset.")
    parser.add_argument("--size", type=int, default=5000, help="Number of synthetic rows to generate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_materials_large.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = build_dataset(root=root, target_size=max(300, args.size))
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Generated {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
