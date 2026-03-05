def evidence_bonus(context_text: str) -> int:
    t = context_text.lower()
    bonus = 0
    if any(k in t for k in ["renewable", "carbon-free", "scope 1", "scope 2", "scope 3", "emissions"]):
        bonus += 1
    if any(k in t for k in ["board", "committee", "oversight"]):
        bonus += 1
    if any(k in t for k in ["assurance", "audit", "audited"]):
        bonus += 1
    return bonus