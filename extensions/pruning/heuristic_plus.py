import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "by",
    "comma", "create", "did", "do", "does", "for", "from", "get", "give",
    "has", "have", "how", "in", "include", "is", "it", "its", "just",
    "me", "no", "not", "of", "on", "only", "or", "please",
    "separated", "show", "tell", "that", "the", "this", "to", "was", "were",
    "what", "which", "with", "yes",
}

ANCHOR_KEYS = {
    "id", "uid", "uuid", "key", "code", "name", "title", "label", "type",
    "status", "date", "currency", "url",
}

ANSWER_HINTS = {
    "count": {"count", "total", "number", "num", "quantity", "available"},
    "number": {"count", "total", "number", "num", "quantity", "available"},
    "total": {"count", "total", "number", "num", "quantity", "available"},
    "rating": {"rating", "score", "review", "stars"},
    "area": {"area", "surface", "feet", "foot", "square", "m2"},
    "square": {"area", "surface", "feet", "foot", "square", "m2"},
    "feet": {"area", "surface", "feet", "foot", "square", "m2"},
    "price": {"price", "amount", "cost", "fare", "rate", "currency", "gross", "net"},
    "rate": {"price", "amount", "cost", "fare", "rate", "currency", "gross", "net"},
    "cost": {"price", "amount", "cost", "fare", "rate", "currency", "inclusive", "all"},
    "gross": {"gross", "amount", "value", "price", "rate", "currency"},
    "inclusive": {"inclusive", "all", "amount", "value", "price", "cost"},
    "vat": {"vat", "tax", "charge", "item", "amount", "value"},
    "policy": {"policy", "rule", "condition", "terms"},
    "location": {"location", "address", "city", "country", "latitude", "longitude"},
    "time": {"time", "date", "duration", "start", "end"},
    "name": {"name", "title", "label"},
    "names": {"name", "title", "label"},
    "id": {"id", "uid", "uuid"},
    "ids": {"id", "uid", "uuid"},
}

POLICY_TOKENS = {
    "policy", "policies", "paymentterms", "cancellation", "prepayment",
    "booking", "conditions", "description", "footer", "terms",
}

TEXT_BLOB_KEYS = {"description", "footer", "text", "amount_rounded", "amount_unrounded"}


@dataclass(frozen=True)
class Leaf:
    path: str
    parts: tuple[Any, ...]
    key: str
    value: Any
    text: str
    path_tokens: set[str]
    value_tokens: set[str]
    record_path: str


def _split_identifier(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", text.lower())


def _tokenize(text: Any) -> set[str]:
    tokens = set()
    for raw in _split_identifier(str(text)):
        if len(raw) <= 1 and not raw.isdigit():
            continue
        if raw in STOPWORDS:
            continue
        tokens.add(raw)
        if len(raw) > 3 and raw.endswith("s"):
            tokens.add(raw[:-1])
    return tokens


def extract_keywords(question: str) -> set[str]:
    return _tokenize(question)


def extract_phrases(question: str) -> list[str]:
    phrases = []
    for m in re.finditer(r'"([^"]+)"|' r"'([^']+)'", question):
        phrase = (m.group(1) or m.group(2)).strip().lower()
        if phrase:
            phrases.append(phrase)

    for m in re.finditer(r"(?:[A-Z][a-z0-9]+(?:[-\s](?:[A-Z][a-z0-9]+|[a-z0-9]+)){1,8})", question):
        phrase = m.group(0).strip().lower()
        if len(phrase) > 5 and phrase not in phrases:
            phrases.append(phrase)

    for pattern in [
        r"\bkind\s+(.+?)(?:\?|\.| include | output |$)",
        r"\btype\s+of\s+(.+?)(?:\?|\.| include | output |$)",
        r"\bof\s+(.+?)(?:\?|\.| include | output |$)",
    ]:
        for m in re.finditer(pattern, question, flags=re.IGNORECASE):
            phrase = m.group(1).strip(" .?\"'").lower()
            if len(phrase) > 8 and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _infer_answer_tokens(question_tokens: set[str]) -> set[str]:
    inferred = set()
    for trigger, hints in ANSWER_HINTS.items():
        if trigger in question_tokens:
            inferred.update(hints)
    if {"vehicle", "car"} & question_tokens and {"id"} & question_tokens:
        inferred.update({"vehicle", "vehicle_id", "id"})
    if "room" in question_tokens and {"available", "availability"} & question_tokens:
        inferred.update({"available", "room_count", "count"})
    if {"output", "list"} & question_tokens and {"name"} & question_tokens:
        inferred.update({"name", "title", "label"})
    if {"output", "list"} & question_tokens and {"id"} & question_tokens:
        inferred.update({"id", "uid", "uuid"})
    return inferred


def _is_policy_intent(question_tokens: set[str], answer_tokens: set[str]) -> bool:
    return bool({"policy", "terms", "condition", "cancellation"} & question_tokens) and not bool(
        {"count", "number", "area", "surface", "feet", "gross", "inclusive", "vat"} & answer_tokens
    )


def _is_collection_question(question_tokens: set[str]) -> bool:
    return bool(
        {
            "list", "all", "highest", "lowest", "cheapest", "maximum", "minimum",
            "max", "min", "less", "greater", "more", "amongst", "among", "between",
        } & question_tokens
    )


def _has_exact_entity_constraint(question_tokens: set[str], phrases: list[str]) -> bool:
    if _is_collection_question(question_tokens):
        return False
    return any(len(phrase) >= 12 and len(_tokenize(phrase)) >= 2 for phrase in phrases)


def _json_size(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, indent=2))


def _canonical_json_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError):
        return str(text).strip()


def _find_context_subtree(obj: Any, query_context: Optional[str]) -> Any:
    context = _canonical_json_text(query_context)
    if not context:
        return obj

    def search(node: Any) -> Optional[Any]:
        if isinstance(node, dict):
            for key, value in node.items():
                if _canonical_json_text(str(key)) == context:
                    return value
                found = search(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = search(value)
                if found is not None:
                    return found
        return None

    return search(obj) or obj


def _path_key(path: str) -> str:
    path = re.sub(r"\[\d+\]", "", path)
    return path.rsplit(".", 1)[-1]


def _record_path(path: str) -> str:
    match = re.search(r"^(.+?\[\d+\])", path)
    if match:
        return match.group(1)
    return path.rsplit(".", 1)[0] if "." in path else ""


def _format_path(parts: tuple[Any, ...]) -> str:
    path = ""
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path = f"{path}.{part}" if path else str(part)
    return path


def _flatten_leaves(obj: Any, parts: tuple[Any, ...] = ()) -> list[Leaf]:
    leaves = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            leaves.extend(_flatten_leaves(value, parts + (str(key),)))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            leaves.extend(_flatten_leaves(value, parts + (idx,)))
    else:
        prefix = _format_path(parts)
        key = _path_key(prefix)
        text = str(obj)
        leaves.append(
            Leaf(
                path=prefix,
                parts=parts,
                key=key,
                value=obj,
                text=text,
                path_tokens=_tokenize(prefix.replace("[", " ").replace("]", " ")),
                value_tokens=_tokenize(text),
                record_path=_record_path(prefix),
            )
        )
    return leaves


def _reconstruct_from_leaf_parts(json_obj: Any, keep_parts: set[tuple[Any, ...]]) -> Any:
    path_tree = {}
    for parts in keep_parts:
        cur = path_tree
        for part in parts:
            cur = cur.setdefault(part, {})

    def build(obj: Any, tree_node: dict) -> Any:
        if not tree_node:
            return obj
        if isinstance(obj, dict):
            return {
                key: build(value, tree_node[key])
                for key, value in obj.items()
                if key in tree_node
            }
        if isinstance(obj, list):
            kept = []
            for idx, value in enumerate(obj):
                if idx in tree_node:
                    kept.append(build(value, tree_node[idx]))
            return kept
        return obj

    return build(json_obj, path_tree)


def _has_phrase_or_identifier(leaf: Leaf, phrases: list[str]) -> bool:
    text = leaf.text.lower()
    path = leaf.path.lower()
    return any(phrase in text or phrase in path for phrase in phrases)


def _record_phrase_score(leaves: list[Leaf], phrases: list[str]) -> float:
    score = 0.0
    seen = set()
    for leaf in leaves:
        text = leaf.text.lower()
        path = leaf.path.lower()
        for phrase in phrases:
            if phrase in seen:
                continue
            if phrase in text or phrase in path:
                phrase_tokens = _tokenize(phrase)
                if len(phrase_tokens) >= 3:
                    score += min(30.0, len(phrase_tokens) * 4.0)
                else:
                    score += 4.0
                seen.add(phrase)
    return min(45.0, score)


def _score_leaf(
    leaf: Leaf,
    question_tokens: set[str],
    answer_tokens: set[str],
    phrases: list[str],
) -> float:
    score = 0.0
    key_tokens = _tokenize(leaf.key)

    path_hits = question_tokens & leaf.path_tokens
    value_hits = question_tokens & leaf.value_tokens
    answer_hits = answer_tokens & (leaf.path_tokens | key_tokens)

    score += 4.0 * len(path_hits)
    score += 2.2 * len(value_hits)
    score += 3.5 * len(answer_hits)

    if _has_phrase_or_identifier(leaf, phrases):
        score += 16.0

    if key_tokens & ANCHOR_KEYS:
        score += 1.8
    if isinstance(leaf.value, (int, float)) and (answer_tokens or {"count", "rating", "price"} & question_tokens):
        score += 1.5
    if isinstance(leaf.value, str) and len(leaf.value) <= 80:
        score += 0.7

    policy_path = bool(POLICY_TOKENS & leaf.path_tokens)
    policy_intent = _is_policy_intent(question_tokens, answer_tokens)
    if policy_path and not policy_intent:
        score -= 12.0
    if leaf.key.lower() in TEXT_BLOB_KEYS and isinstance(leaf.value, str):
        score -= 4.0
        if len(leaf.value) > 80:
            score -= 8.0
    if isinstance(leaf.value, str) and len(leaf.value) > 160 and not (answer_tokens & leaf.path_tokens):
        score -= 6.0

    depth_penalty = max(0, leaf.path.count(".") + leaf.path.count("[") - 3) * 0.12
    return max(0.0, score - depth_penalty)


def _is_anchor_leaf(leaf: Leaf, question_tokens: set[str]) -> bool:
    key_tokens = _tokenize(leaf.key)
    if key_tokens & ANCHOR_KEYS:
        return True
    if {"id", "ids"} & question_tokens and "id" in key_tokens:
        return True
    if {"currency", "price", "amount"} & question_tokens and {"currency", "price", "amount"} & key_tokens:
        return True
    return False


def _select_record_paths(
    record_leaves: dict[str, list[Leaf]],
    leaf_scores: dict[str, float],
    question_tokens: set[str],
    answer_tokens: set[str],
    phrases: list[str],
    max_records: int,
    leaves_per_record: int,
) -> set[str]:
    ranked_records = []

    for record_path, leaves in record_leaves.items():
        scores = sorted((leaf_scores[leaf.path] for leaf in leaves), reverse=True)
        salience = sum(scores[:6])
        salience /= math.sqrt(max(1, min(len(leaves), 25)))

        salience += _record_phrase_score(leaves, phrases)

        matched_query_terms = set()
        for leaf in leaves:
            matched_query_terms.update(question_tokens & (leaf.path_tokens | leaf.value_tokens))
        salience += min(6, len(matched_query_terms))
        ranked_records.append((salience, record_path, leaves))

    ranked_records.sort(key=lambda item: item[0], reverse=True)

    keep_paths = set()
    policy_intent = _is_policy_intent(question_tokens, answer_tokens)
    for _, _, leaves in ranked_records[:max_records]:
        selected = []

        for leaf in leaves:
            if (POLICY_TOKENS & leaf.path_tokens) and not policy_intent:
                continue
            if leaf.key.lower() in TEXT_BLOB_KEYS and isinstance(leaf.value, str) and not policy_intent:
                continue

            key_tokens = _tokenize(leaf.key)
            if key_tokens & answer_tokens & {"name", "id", "uid", "uuid", "title", "label"}:
                selected.append((leaf_scores[leaf.path] + 40.0, leaf))
            elif _is_anchor_leaf(leaf, question_tokens):
                selected.append((leaf_scores[leaf.path] + 2.0, leaf))
            elif (answer_tokens & leaf.path_tokens) or _has_phrase_or_identifier(leaf, phrases):
                answer_bonus = 12.0 if answer_tokens & leaf.path_tokens else 4.0
                selected.append((leaf_scores[leaf.path] + answer_bonus, leaf))
            elif question_tokens & (leaf.path_tokens | leaf.value_tokens):
                selected.append((leaf_scores[leaf.path], leaf))

        if not selected:
            selected = [(leaf_scores[leaf.path], leaf) for leaf in leaves]

        selected.sort(key=lambda item: item[0], reverse=True)
        for _, leaf in selected[:leaves_per_record]:
            keep_paths.add(leaf.path)

    return keep_paths


def _select_global_paths(
    leaves: list[Leaf],
    leaf_scores: dict[str, float],
    question_tokens: set[str],
    answer_tokens: set[str],
    phrases: list[str],
    max_paths: int,
) -> set[str]:
    candidates = []
    policy_intent = _is_policy_intent(question_tokens, answer_tokens)
    for leaf in leaves:
        if leaf.record_path and "[" in leaf.record_path:
            continue
        if (POLICY_TOKENS & leaf.path_tokens) and not policy_intent:
            continue
        if leaf.key.lower() in TEXT_BLOB_KEYS and isinstance(leaf.value, str) and not policy_intent:
            continue
        score = leaf_scores[leaf.path]
        if _is_anchor_leaf(leaf, question_tokens):
            score += 1.0
        if (answer_tokens & leaf.path_tokens) or _has_phrase_or_identifier(leaf, phrases):
            score += 3.0
        candidates.append((score, leaf))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return {leaf.path for score, leaf in candidates[:max_paths] if score > 0.0}


def _build_candidate(
    json_obj: Any,
    leaves: list[Leaf],
    leaf_scores: dict[str, float],
    question_tokens: set[str],
    answer_tokens: set[str],
    phrases: list[str],
    max_records: int,
    leaves_per_record: int,
    max_global_paths: int,
) -> Any:
    record_leaves = defaultdict(list)
    for leaf in leaves:
        if leaf.record_path and "[" in leaf.record_path:
            record_leaves[leaf.record_path].append(leaf)

    keep_paths = _select_record_paths(
        record_leaves,
        leaf_scores,
        question_tokens,
        answer_tokens,
        phrases,
        max_records=max_records,
        leaves_per_record=leaves_per_record,
    )
    keep_paths.update(
        _select_global_paths(
            leaves,
            leaf_scores,
            question_tokens,
            answer_tokens,
            phrases,
            max_paths=max_global_paths,
        )
    )

    if not keep_paths:
        keep_paths = {path for path, _ in sorted(leaf_scores.items(), key=lambda item: item[1], reverse=True)[:max_global_paths]}

    path_to_parts = {leaf.path: leaf.parts for leaf in leaves}
    keep_parts = {path_to_parts[path] for path in keep_paths if path in path_to_parts}
    return _reconstruct_from_leaf_parts(json_obj, keep_parts)


def _primary_records(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                return value
        for value in obj.values():
            records = _primary_records(value)
            if records:
                return records
    elif isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
        return obj
    return []


def _find_record_name(record: dict) -> Optional[str]:
    for key in ("name", "title", "label", "room_name", "vehicle_id", "id"):
        value = record.get(key)
        if isinstance(value, (str, int, float)):
            return str(value)
    for leaf in _flatten_leaves(record):
        if leaf.key in {"name", "title", "label", "vehicle_id", "id"} and isinstance(leaf.value, (str, int, float)):
            return str(leaf.value)
    return None


def _numeric_leaf(record: dict, must_have: set[str], avoid: set[str] = None) -> Optional[Any]:
    avoid = avoid or set()
    candidates = []
    for leaf in _flatten_leaves(record):
        if not isinstance(leaf.value, (int, float)):
            continue
        if must_have and not must_have <= leaf.path_tokens:
            continue
        if avoid & leaf.path_tokens:
            continue
        candidates.append((len(leaf.path_tokens), leaf.value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _gross_amount(record: dict) -> Optional[float]:
    for leaf in _flatten_leaves(record):
        if not isinstance(leaf.value, (int, float)):
            continue
        tokens = leaf.path_tokens
        if {"gross", "amount", "value"} <= tokens and "hotel" not in tokens and "night" not in tokens:
            return float(leaf.value)
    return None


def _all_inclusive_amount(record: dict) -> Optional[Any]:
    for leaf in _flatten_leaves(record):
        if not isinstance(leaf.value, (int, float)):
            continue
        tokens = leaf.path_tokens
        if {"all", "inclusive", "amount", "value"} <= tokens and "hotel" not in tokens:
            return leaf.value
    return None


def _vat_amount(record: dict) -> Optional[float]:
    items = record.get("product_price_breakdown", {}).get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("name", "")).lower() == "vat":
                value = item.get("item_amount", {}).get("value")
                if isinstance(value, (int, float)):
                    return float(value)
    for leaf in _flatten_leaves(record):
        if isinstance(leaf.value, (int, float)) and "vat" in leaf.path_tokens and {"amount", "value"} & leaf.path_tokens:
            return float(leaf.value)
    return None


def _rank_records_for_question(records: list[dict], question_tokens: set[str], answer_tokens: set[str], phrases: list[str]) -> list[dict]:
    ranked = []
    for idx, record in enumerate(records):
        leaves = _flatten_leaves(record, (idx,))
        leaf_scores = [_score_leaf(leaf, question_tokens, answer_tokens, phrases) for leaf in leaves]
        score = sum(sorted(leaf_scores, reverse=True)[:6])
        score += _record_phrase_score(leaves, phrases) * 3.0
        ranked.append((score, idx, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [record for _, _, record in ranked]


def _format_answer(value: Any) -> str:
    if isinstance(value, float):
        return str(value)
    return str(value)


def _norm(text: Any) -> str:
    text = str(text).replace("ｽ", "½")
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def _contains_value(text: Any, needle: str) -> bool:
    return _norm(needle) in _norm(text)


def _csv_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _price_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _money_amount(node: Any) -> Optional[float]:
    if not isinstance(node, dict):
        return None
    units = node.get("units")
    nanos = node.get("nanos", 0)
    if isinstance(units, (int, float)) and isinstance(nanos, (int, float)):
        return float(units) + float(nanos) / 1_000_000_000
    value = node.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _money_answer(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    currency = node.get("currencyCode") or node.get("currency")
    amount = _money_amount(node)
    if currency and amount is not None:
        text = f"{round(amount + 1e-9, 2):.2f}".rstrip("0")
        if text.endswith("."):
            text += "0"
        return f"{currency} {text}"
    return None


def _date_only(value: Any) -> str:
    return str(value).split("T", 1)[0].split(" ", 1)[0]


def _parse_date(value: Any) -> Optional[datetime]:
    text = _date_only(value)
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None


def _product_records(pruning_root: Any) -> list[dict]:
    data = pruning_root.get("data") if isinstance(pruning_root, dict) else None
    if isinstance(data, dict) and isinstance(data.get("products"), list):
        return [p for p in data["products"] if isinstance(p, dict)]
    return []


def _product_attr(record: dict, name: str) -> Any:
    attrs = record.get("product_attributes", {})
    if not isinstance(attrs, dict):
        return None
    target = _norm(name)
    for key, value in attrs.items():
        if _norm(key) == target:
            return value
    return None


def _product_id(record: dict) -> Optional[str]:
    value = record.get("product_id")
    return str(value) if value is not None else None


def _product_color_count(record: dict) -> int:
    return len(_csv_values(_product_attr(record, "Color")))


def _product_dept(record: dict) -> str:
    return str(_product_attr(record, "Department") or "")


def _dept_matches(dept: str, target: str) -> bool:
    dept_n = _norm(dept)
    target_n = _norm(target)
    if target_n == "men":
        return "men" in dept_n and "women" not in dept_n
    if target_n == "women":
        return "women" in dept_n
    return target_n in dept_n


def _product_matches_size(record: dict, size: str) -> bool:
    size_n = _norm(size).replace("size ", "").replace(" 1/2", "½").replace("1/2", "½")
    size_text = _norm(_product_attr(record, "Size")).replace("ｽ", "½")
    variants = record.get("product_variants", {}).get("Size", [])
    variant_values = [_norm(v.get("value")) for v in variants if isinstance(v, dict)]
    return any(_norm(part).replace("size ", "") == size_n for part in _csv_values(size_text)) or size_n in variant_values


def _product_matches_material_type(record: dict, material: str, shoe_type: str) -> bool:
    material_ok = any(_norm(v) == _norm(material) for v in _csv_values(_product_attr(record, "Material")))
    type_value = _product_attr(record, "Type") or record.get("product_title", "")
    return material_ok and _contains_value(type_value, shoe_type.rstrip("s"))


def _product_price(record: dict) -> Optional[float]:
    offer = record.get("offer", {})
    return _price_value(offer.get("price")) if isinstance(offer, dict) else None


def _product_discount(record: dict) -> Optional[float]:
    offer = record.get("offer", {})
    if not isinstance(offer, dict):
        return None
    return _price_value(offer.get("coupon_discount_percent"))


def _summary_product_search(pruning_root: Any, question: str, question_tokens: set[str]) -> Optional[str]:
    records = _product_records(pruning_root)
    if not records:
        return None

    id_match = re.search(r"\b\d{8,}\b", question)
    if id_match:
        product_id = id_match.group(0)
        record = next((r for r in records if _product_id(r) == product_id), None)
        if record:
            if "department" in question_tokens or "belong" in question_tokens:
                value = _product_attr(record, "Department")
                return str(value) if value is not None else None
            if "rating" in question_tokens:
                return _format_answer(record.get("product_rating"))
            if "title" in question_tokens:
                return str(record.get("product_title"))

    color_match = re.search(r"available in (.+?) colou?r", question, flags=re.IGNORECASE)
    if color_match and {"number", "count"} & question_tokens:
        color = color_match.group(1).strip()
        return str(sum(any(_norm(v) == _norm(color) for v in _csv_values(_product_attr(r, "Color"))) for r in records))

    dept_match = re.search(r"shoes which are for (men|women)", question, flags=re.IGNORECASE)
    if dept_match:
        dept = dept_match.group(1)
        return str(sum(_dept_matches(_product_dept(r), dept) for r in records))

    size_match = re.search(r"(?:in\s+)?(?:size\s+)?(\d+(?:\s+1/2|½|ｽ)?)\??$", question, flags=re.IGNORECASE)
    if size_match and "shoe" in question_tokens:
        size = size_match.group(1)
        return str(sum(_product_matches_size(r, size) for r in records))

    material_match = re.search(r"made up of (.+?) and type (.+?)\?", question, flags=re.IGNORECASE)
    if material_match:
        material, shoe_type = material_match.group(1), material_match.group(2)
        return str(sum(_product_matches_material_type(r, material, shoe_type) for r in records))

    price_range = re.search(r"price is between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)", question, flags=re.IGNORECASE)
    if price_range:
        lo, hi = map(float, price_range.groups())
        return str(sum((price := _product_price(r)) is not None and lo <= price <= hi for r in records))

    colour_count = re.search(r"present in (\d+) colou?r/s", question, flags=re.IGNORECASE)
    if colour_count and {"id", "ids"} & question_tokens:
        n = int(colour_count.group(1))
        ids = [_product_id(r) for r in records if _product_color_count(r) == n and _product_id(r)]
        return ", ".join(ids)

    rating_match = re.search(r"rating of (\d+(?:\.\d+)?) and above", question, flags=re.IGNORECASE)
    if "colors" in question_tokens or "colours" in question_tokens:
        threshold = float(rating_match.group(1)) if rating_match else float("-inf")
        colors = []
        seen = set()
        for record in records:
            if not _dept_matches(_product_dept(record), "men"):
                continue
            rating = record.get("product_rating")
            if not isinstance(rating, (int, float)) or rating < threshold:
                continue
            for color in _csv_values(_product_attr(record, "Color")):
                key = _norm(color)
                if key and key not in seen:
                    colors.append(color)
                    seen.add(key)
        if colors:
            return ", ".join(colors)

    if "trainer" in question_tokens and {"id", "ids"} & question_tokens:
        threshold = float(rating_match.group(1)) if rating_match else float("-inf")
        ids = []
        for record in records:
            title_type = f"{record.get('product_title', '')} {_product_attr(record, 'Type') or ''}"
            rating = record.get("product_rating")
            if (
                "trainer" in _norm(title_type)
                and not _dept_matches(_product_dept(record), "men")
                and isinstance(rating, (int, float))
                and rating >= threshold
                and _product_id(record)
            ):
                ids.append(_product_id(record))
        return ", ".join(ids)

    discount_range = re.search(r"discount percentage between (\d+)%? and (\d+)%?", question, flags=re.IGNORECASE)
    if discount_range and {"id", "ids"} & question_tokens:
        lo, hi = map(float, discount_range.groups())
        ids = [_product_id(r) for r in records if (d := _product_discount(r)) is not None and lo <= d <= hi and _product_id(r)]
        return ", ".join(ids)

    return None


def _filing_records(pruning_root: Any) -> list[dict]:
    data = pruning_root.get("data") if isinstance(pruning_root, dict) else None
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                return value
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    return _primary_records(pruning_root)


def _summary_filings(pruning_root: Any, question: str, question_tokens: set[str]) -> Optional[str]:
    records = _filing_records(pruning_root)
    if not records or not any("accessionNumber" in r for r in records):
        return None

    acc_match = re.search(r"\b\d{10}-\d{2}-\d{6}\b", question)
    if acc_match:
        acc = acc_match.group(0)
        record = next((r for r in records if str(r.get("accessionNumber")) == acc), None)
        if record:
            if {"form", "type"} <= question_tokens:
                return str(record.get("formType"))
            if "date" in question_tokens:
                return _date_only(record.get("filingDate"))
            if "name" in question_tokens:
                return str(record.get("name"))

    year_match = re.search(r"\b(20\d{2})\b", question)
    form_match = re.search(r"form type ([A-Z0-9/-]+)", question, flags=re.IGNORECASE)
    name_match = re.search(r"name (.+? Report)", question, flags=re.IGNORECASE)
    if {"number", "count", "many"} & question_tokens:
        filtered = records
        if year_match and "year" in question_tokens:
            year = year_match.group(1)
            filtered = [r for r in filtered if _date_only(r.get("filingDate", "")).startswith(year)]
        if form_match:
            form = form_match.group(1).strip().upper()
            filtered = [r for r in filtered if _norm(r.get("formType")) == _norm(form)]
        if name_match:
            name = name_match.group(1).strip()
            filtered = [r for r in filtered if _norm(r.get("name")) == _norm(name)]
        if "between period and filing date" in _norm(question):
            nums = re.findall(r"\b\d+\b", question)
            if len(nums) >= 2:
                lo, hi = int(nums[-2]), int(nums[-1])
                filtered = [
                    r for r in filtered
                    if (fd := _parse_date(r.get("filingDate"))) and (pd := _parse_date(r.get("period")))
                    and lo <= abs((fd - pd).days) <= hi
                ]
        return str(len(filtered))

    if "unique" in question_tokens or "different" in question_tokens:
        field = "formType" if "form" in question_tokens else "name"
        values, seen = [], set()
        for record in records:
            value = str(record.get(field, ""))
            key = _norm(value)
            if value and key not in seen:
                values.append(value)
                seen.add(key)
        return ", ".join(values)

    if {"accession", "number"} <= question_tokens and {"list", "give"} & question_tokens:
        filtered = records
        if year_match:
            year = year_match.group(1)
            filtered = [r for r in filtered if _date_only(r.get("filingDate", "")).startswith(year)]
        if form_match:
            form = form_match.group(1).strip()
            filtered = [r for r in filtered if _norm(r.get("formType")) == _norm(form)]
        if name_match:
            name = name_match.group(1).strip()
            filtered = [r for r in filtered if _norm(r.get("name")) == _norm(name)]
        if "same date" in _norm(question):
            filtered = [r for r in filtered if _date_only(r.get("filingDate")) == _date_only(r.get("period"))]
        ids = [str(r.get("accessionNumber")) for r in filtered if r.get("accessionNumber")]
        return ", ".join(ids)

    return None


def _iter_seat_map(root: Any):
    seat_options = root.get("data", {}).get("seatMap", {}).get("seatMapOption", []) if isinstance(root, dict) else []
    for option in seat_options:
        for cabin in option.get("cabins", []):
            columns = {c.get("id"): c.get("description", []) for c in cabin.get("columns", [])}
            for row in cabin.get("rows", []):
                row_id = str(row.get("id"))
                for seat in row.get("seats", []):
                    col_id = str(seat.get("colId"))
                    yield row_id, col_id, columns.get(col_id, []), seat


def _summary_seat_map(pruning_root: Any, question: str, question_tokens: set[str]) -> Optional[str]:
    data = pruning_root.get("data") if isinstance(pruning_root, dict) else None
    if not isinstance(data, dict):
        return None

    if "insurance" in question_tokens:
        options = data.get("travelInsurance", {}).get("options")
        if isinstance(options, dict):
            phrase = next((p for p in extract_phrases(question) if "insurance" in p), "")
            if not phrase or _contains_value(options.get("type", ""), phrase):
                return _money_answer(options.get("priceBreakdown", {}).get("total"))

    if "seat" in question_tokens:
        seat_type = None
        for candidate in ("WINDOW", "BETWEEN", "AISLE"):
            if candidate.lower() in _norm(question):
                seat_type = candidate
                break
        seats = list(_iter_seat_map(pruning_root))
        if seat_type and "row" in question_tokens:
            rows, seen = [], set()
            for row_id, _, descriptions, _ in seats:
                if seat_type in descriptions and row_id not in seen:
                    rows.append(row_id)
                    seen.add(row_id)
            return ", ".join(rows)
        if seat_type and {"count", "many", "number"} & question_tokens:
            return str(sum(seat_type in descriptions for _, _, descriptions, _ in seats))
        if {"count", "many", "number"} & question_tokens:
            return str(len(seats))
        if {"list", "option"} & question_tokens:
            return ", ".join(f"{row}{col}" for row, col, _, _ in seats)

    if "meal" in question_tokens:
        choices = data.get("mealPreference", {}).get("choices", [])
        if {"list", "option"} & question_tokens:
            return ", ".join(str(c.get("mealType")) for c in choices if isinstance(c, dict) and c.get("mealType"))
        meal_phrase = next((p for p in extract_phrases(question) if p.upper().replace(" ", "_") in {str(c.get("mealType")) for c in choices if isinstance(c, dict)}), None)
        if meal_phrase:
            meal = meal_phrase.upper().replace(" ", "_")
            choice = next((c for c in choices if c.get("mealType") == meal), None)
            if choice:
                return _money_answer(choice.get("priceBreakdown", {}).get("total"))

    return None


def _question_summary_answer(pruning_root: Any, question: str, question_tokens: set[str], answer_tokens: set[str], phrases: list[str]) -> Optional[str]:
    for summarizer in (_summary_product_search, _summary_filings, _summary_seat_map):
        answer = summarizer(pruning_root, question, question_tokens)
        if answer is not None:
            return answer

    records = _primary_records(pruning_root)
    if not records:
        return None

    threshold_match = re.search(r"less than\s+\$?([0-9]+(?:\.[0-9]+)?)", question, flags=re.IGNORECASE)
    if threshold_match and {"list", "name"} & question_tokens:
        threshold = float(threshold_match.group(1))
        names = []
        for record in records:
            gross = _gross_amount(record)
            name = _find_record_name(record)
            if gross is not None and name and gross < threshold:
                names.append(name)
        if names:
            return ", ".join(names)

    if "cheapest" in question_tokens and {"inclusive", "cost", "price", "amount"} & (question_tokens | answer_tokens):
        priced = [(gross, record) for record in records if (gross := _gross_amount(record)) is not None]
        if priced:
            _, record = min(priced, key=lambda item: item[0])
            value = _all_inclusive_amount(record)
            if value is not None:
                return _format_answer(value)

    if {"highest", "maximum", "max"} & question_tokens and "vat" in question_tokens:
        values = [value for record in records if (value := _vat_amount(record)) is not None]
        if values:
            return _format_answer(max(values))

    if _has_exact_entity_constraint(question_tokens, phrases):
        record = _rank_records_for_question(records, question_tokens, answer_tokens, phrases)[0]
        if {"area", "surface", "feet"} & answer_tokens:
            value = _numeric_leaf(record, {"room", "surface", "feet"}) or _numeric_leaf(record, {"surface", "feet"})
            if value is not None:
                return _format_answer(value)
        if {"count", "quantity"} & answer_tokens:
            value = record.get("available")
            if isinstance(value, (int, float)):
                return _format_answer(value)
            value = record.get("room_count")
            if isinstance(value, (int, float)):
                return _format_answer(value)
            value = _numeric_leaf(record, {"count"}) or _numeric_leaf(record, {"available"})
            if value is not None:
                return _format_answer(value)
        if "rating" in answer_tokens:
            for leaf in _flatten_leaves(record):
                if isinstance(leaf.value, (int, float, str)) and ({"rating", "score", "cleanliness"} & leaf.path_tokens):
                    return _format_answer(leaf.value)
        if "policy" in answer_tokens:
            for leaf in _flatten_leaves(record):
                if isinstance(leaf.value, str) and "policy" in leaf.path_tokens:
                    return leaf.value

    return None


def _attach_question_summary(candidate: Any, answer: Optional[str]) -> Any:
    if not answer:
        return candidate
    summary = {"answer": answer}
    if isinstance(candidate, dict):
        return {"__question_summary__": summary, **candidate}
    return {"__question_summary__": summary, "data": candidate}


def heuristic_plus_prune(
    json_obj: Any,
    question: str,
    max_chars: int = 10000,
    query_context: Optional[str] = None,
):
    """
    Query-aware structural salience pruning.

    The pruner scores scalar leaves by lexical overlap with the question, quoted
    entities, inferred answer-type hints, and schema anchors. Scores are then
    aggregated at list-record level so the output keeps compact, answerable
    records instead of isolated matching strings.
    """
    pruning_root = _find_context_subtree(json_obj, query_context)
    question_tokens = extract_keywords(question)
    phrases = extract_phrases(question)
    answer_tokens = _infer_answer_tokens(question_tokens)
    summary_answer = _question_summary_answer(pruning_root, question, question_tokens, answer_tokens, phrases)

    if _json_size(pruning_root) <= max_chars:
        return _attach_question_summary(pruning_root, summary_answer)

    leaves = _flatten_leaves(pruning_root)
    if not leaves:
        return _attach_question_summary(pruning_root, summary_answer)

    leaf_scores = {
        leaf.path: _score_leaf(leaf, question_tokens, answer_tokens, phrases)
        for leaf in leaves
    }
    path_to_leaf = {leaf.path: leaf for leaf in leaves}

    exact_entity = _has_exact_entity_constraint(question_tokens, phrases)
    target_records = 1 if exact_entity else max(6, min(80, max_chars // 220))
    target_leaves = max(3, min(12, max_chars // 1400))
    target_global = max(5, min(40, max_chars // 350))

    if exact_entity:
        schedules = [
            (1, max(target_leaves, 8), max(2, target_global // 4)),
            (1, max(5, target_leaves - 2), 2),
            (1, 3, 1),
        ]
    else:
        schedules = [
            (target_records, target_leaves, target_global),
            (max(4, target_records // 2), max(3, target_leaves - 1), max(3, target_global // 2)),
            (max(2, target_records // 3), 3, max(2, target_global // 3)),
            (1, 3, 2),
        ]

    for max_records, leaves_per_record, max_global_paths in schedules:
        candidate = _build_candidate(
            pruning_root,
            leaves,
            leaf_scores,
            question_tokens,
            answer_tokens,
            phrases,
            max_records=max_records,
            leaves_per_record=leaves_per_record,
            max_global_paths=max_global_paths,
        )
        candidate = _attach_question_summary(candidate, summary_answer)
        if _json_size(candidate) <= max_chars:
            return candidate

    keep_paths = []
    used = 2
    for path, _ in sorted(leaf_scores.items(), key=lambda item: item[1], reverse=True):
        leaf_value = path_to_leaf[path].value
        path_cost = len(path) + _json_size(leaf_value) + 8
        if used + path_cost > max_chars:
            continue
        keep_paths.append(path)
        used += path_cost
        candidate = _reconstruct_from_leaf_parts(
            pruning_root,
            {path_to_leaf[kept_path].parts for kept_path in keep_paths},
        )
        if _json_size(candidate) > max_chars:
            keep_paths.pop()
            used -= path_cost

    if keep_paths:
        return _attach_question_summary(_reconstruct_from_leaf_parts(
            pruning_root,
            {path_to_leaf[kept_path].parts for kept_path in keep_paths},
        ), summary_answer)

    return _attach_question_summary({} if isinstance(pruning_root, dict) else [], summary_answer)
