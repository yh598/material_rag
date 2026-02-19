import argparse
import json
from itertools import product
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


def all_filter_values(products: list[dict]) -> tuple[list[str], list[str], list[str]]:
    divisions = sorted({str(p.get("division", "")).strip() for p in products if str(p.get("division", "")).strip()})
    departments = sorted({str(p.get("department", "")).strip() for p in products if str(p.get("department", "")).strip()})
    categories = sorted({str(p.get("category", "")).strip() for p in products if str(p.get("category", "")).strip()})
    return divisions, departments, categories


def all_filter_combos(products: list[dict]) -> list[tuple[str, str, str]]:
    divisions, departments, categories = all_filter_values(products)
    return [(d, dep, cat) for d, dep, cat in product(divisions, departments, categories)]


def rows_have_full_combo_coverage(rows: list[dict], combos: list[tuple[str, str, str]]) -> bool:
    if not rows:
        return False
    if not combos:
        return True

    present = set()
    for row in rows:
        combo = (
            str(row.get("division", "")).strip(),
            str(row.get("department", "")).strip(),
            str(row.get("category", "")).strip(),
        )
        if all(combo):
            present.add(combo)

    return all(combo in present for combo in combos)


def _code_token(text: str, fallback: str) -> str:
    cleaned = "".join(ch for ch in str(text).upper() if ch.isalnum())
    if not cleaned:
        return fallback
    return cleaned[:4]


def build_dataset(root: Path, target_size: int, min_per_combo: int = 6) -> list[dict]:
    materials = read_json(root / "materials.json")
    products = read_json(root / "products.json")
    links = read_json(root / "product_materials.json")
    if not materials or not products:
        return []

    combos = all_filter_combos(products)
    min_per_combo = max(1, safe_int(min_per_combo, 6))
    target_size = max(300, safe_int(target_size, 3000), len(combos) * min_per_combo)

    products_by_id = {safe_int(p["id"]): p for p in products}
    materials_by_id = {safe_int(m["id"]): m for m in materials}

    products_by_combo: dict[tuple[str, str, str], list[dict]] = {}
    for p in products:
        combo = (
            str(p.get("division", "")).strip(),
            str(p.get("department", "")).strip(),
            str(p.get("category", "")).strip(),
        )
        if all(combo):
            products_by_combo.setdefault(combo, []).append(p)

    known_consumers = sorted(
        {str(p.get("targetConsumer", "")).strip() for p in products if str(p.get("targetConsumer", "")).strip()}
    ) or ["Unisex"]
    known_seasons = sorted({str(p.get("season", "")).strip() for p in products if str(p.get("season", "")).strip()}) or ["SP25"]

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

    generated: list[dict] = []
    seen_ids = set()
    combo_counts: dict[tuple[str, str, str], int] = {}
    combo_peak_score: dict[tuple[str, str, str], int] = {}

    def choose_product(combo: tuple[str, str, str], seed: int) -> dict:
        candidates = products_by_combo.get(combo, [])
        if candidates:
            return candidates[seed % len(candidates)]

        division, department, category = combo
        synthetic_id = 900000 + (seed % 90000)
        return {
            "id": synthetic_id,
            "code": (
                f"SYN-{_code_token(division, 'DIV')}-"
                f"{_code_token(department, 'DEP')}-{_code_token(category, 'CAT')}"
            ),
            "name": f"{division} {department} {category} Synthetic Product",
            "division": division,
            "department": department,
            "category": category,
            "targetConsumer": pick(known_consumers, seed),
            "season": pick(known_seasons, seed + 7),
        }

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

        division = str(product.get("division", "Performance"))
        department = str(product.get("department", "Running"))
        category = str(product.get("category", "General"))
        combo = (division, department, category)
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
        combo_peak_score[combo] = max(combo_peak_score.get(combo, 0), score)

        generated.append(
            {
                "id": rec_id,
                "name": material.get("name", "Synthetic Material"),
                "type": role,
                "category": category,
                "department": department,
                "division": division,
                "product_id": str(product.get("id", "")),
                "product_code": str(product.get("code", "")),
                "product_name": str(product.get("name", "")),
                "target_consumer": str(product.get("targetConsumer", "")),
                "season": str(product.get("season", "")),
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

    material_count = len(materials)
    if not combos:
        combos = [("Performance", "Running", "General")]

    for combo_idx, combo in enumerate(combos):
        peak_seed = (combo_idx + 1) * 97
        if combo_peak_score.get(combo, 0) < 99:
            product = choose_product(combo, peak_seed)
            material = materials[(peak_seed * 7) % material_count]
            roles = role_candidates(str(material.get("family", "Material")))
            role = roles[peak_seed % len(roles)]
            add_record(product, material, role, 0.99, f"combopeak{combo_idx:03d}")

        while combo_counts.get(combo, 0) < min_per_combo:
            seq = combo_counts.get(combo, 0)
            seed = (combo_idx * 131) + (seq * 17) + 11
            product = choose_product(combo, seed)
            material = materials[(seed * 13 + seq * 3) % material_count]
            roles = role_candidates(str(material.get("family", "Material")))
            role = roles[(seed + seq) % len(roles)]

            score = 0.72 + ((seed + seq * 11) % 24) / 100.0
            if combo[0] == "Performance":
                score += 0.03
            elif combo[0] in {"Outdoor", "Lifestyle"}:
                score += 0.02
            add_record(product, material, role, min(score, 0.97), f"combofill{combo_idx:03d}_{seq:02d}")

    idx = 0
    max_iterations = max(target_size * 8, 2400)
    combo_count = len(combos)
    while len(generated) < target_size and idx < max_iterations:
        combo = combos[(idx * 7 + (idx // 9)) % combo_count]
        product = choose_product(combo, idx + 17)
        material = materials[(idx * 13 + (idx // 7)) % material_count]
        roles = role_candidates(str(material.get("family", "Material")))
        role = roles[(idx + safe_int(product.get("id"))) % len(roles)]

        score = 0.67 + (((idx * 19) + safe_int(product.get("id")) + safe_int(material.get("id"))) % 33) / 100.0
        if combo[0] == "Performance":
            score += 0.05
        elif combo[0] in {"Outdoor", "Lifestyle"}:
            score += 0.03
        add_record(product, material, role, min(score, 0.99), f"syn{idx:05d}")

        if len(generated) < target_size and idx % 3 == 0:
            alt_role = roles[(idx + 1) % len(roles)]
            add_record(product, material, alt_role, max(0.60, score - 0.07), f"alt{idx:05d}")

        idx += 1

    return sorted(generated[:target_size], key=lambda x: x["score"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Generate large synthetic material recommendation dataset.")
    parser.add_argument("--size", type=int, default=12000, help="Number of synthetic rows to generate")
    parser.add_argument(
        "--min-per-combo",
        type=int,
        default=6,
        help="Minimum number of rows for each division/department/category combination",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_materials_large.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = build_dataset(root=root, target_size=max(300, args.size), min_per_combo=max(1, args.min_per_combo))
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    products = read_json(root / "products.json")
    combos = all_filter_combos(products)
    covered = rows_have_full_combo_coverage(rows, combos)
    print(f"Generated {len(rows)} rows -> {output_path}")
    print(f"Filter combinations covered: {covered} ({len(combos)} combinations)")


if __name__ == "__main__":
    main()
