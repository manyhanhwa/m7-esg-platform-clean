import os
import re
import streamlit as st

import pandas as pd
import altair as alt

from openai import OpenAI
from rag import retrieve
from scoring import evidence_bonus

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="M7 ESG Strategy Intelligence", layout="wide")

# ----------------------------
# Secrets / Env (safe)
# ----------------------------
try:
    secret_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    secret_key = None

OPENAI_API_KEY = secret_key or os.getenv("OPENAI_API_KEY")

# 데모 모드에서는 키 없어도 됨 (OpenAI 호출 자체를 안 하니까)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------------------------
# Constants
# ----------------------------
M7 = [
    "NVIDIA (NVDA)", "Microsoft (MSFT)", "Apple (AAPL)",
    "Alphabet (GOOGL)", "Amazon (AMZN)", "Meta (META)", "Tesla (TSLA)"
]

# 데모 모드에서도 “점수/포지셔닝”이 자연스럽게 나오도록 고정값 제공
# (발표 때: A안(데모) → B안(실데이터)로 업그레이드 스토리)
DEMO_SCORE = {
    "NVIDIA (NVDA)": {"E": 76, "S": 72, "G": 78},
    "Microsoft (MSFT)": {"E": 82, "S": 78, "G": 84},
    "Apple (AAPL)": {"E": 79, "S": 76, "G": 82},
    "Alphabet (GOOGL)": {"E": 77, "S": 73, "G": 79},
    "Amazon (AMZN)": {"E": 75, "S": 71, "G": 74},
    "Meta (META)": {"E": 70, "S": 68, "G": 72},
    "Tesla (TSLA)": {"E": 83, "S": 66, "G": 70},
}

DEMO_POSITION = {
    "NVIDIA (NVDA)": (78, 72),
    "Microsoft (MSFT)": (82, 80),
    "Apple (AAPL)": (76, 78),
    "Alphabet (GOOGL)": (74, 73),
    "Amazon (AMZN)": (79, 66),
    "Meta (META)": (70, 60),
    "Tesla (TSLA)": (83, 62),
}

# ----------------------------
# UI Header
# ----------------------------
st.title("🌍 M7 ESG Strategy Intelligence Platform")
st.caption("RAG(근거 기반) + ESG 점수/리스크/포지셔닝 비교 • 데모 모드 지원")

left, right = st.columns([1, 2])

with left:
    companies = st.multiselect("기업 선택", M7, default=["NVIDIA (NVDA)", "Microsoft (MSFT)"])
    detail = st.select_slider("분석 깊이", ["간단", "표준", "심화"], value="표준")
    demo_mode = st.checkbox("데모 모드(결제 없이 확인)", value=True)
    q = st.text_area(
        "요청",
        value="각 기업의 ESG 전략을 E/S/G로 요약하고, 점수/리스크/포지셔닝까지 보여줘."
    )
    run = st.button("분석 실행", type="primary")

with right:
    st.markdown("""
**기능**
- E/S/G 전략 요약(키워드 + 문장)
- ESG Score(E/S/G/Total)
- ESG Risk(E/S/G)
- Positioning(투자강도 × 전략통합)
- 근거(Citations) — *RAG 인덱스 준비 시 자동 표시*
- 📊 랭킹 테이블 + 📍 2D 포지셔닝 지도(4분면/평균선/그룹색)
""")

if not demo_mode and not OPENAI_API_KEY:
    st.warning("데모 모드를 끄면 OpenAI API Key가 필요합니다. (데모 모드는 키 없이도 동작)")

# ----------------------------
# Helpers
# ----------------------------
def company_group(name: str) -> str:
    if "NVIDIA" in name:
        return "AI / Chip"
    if "Microsoft" in name or "Alphabet" in name or "Meta" in name:
        return "Platform"
    if "Apple" in name:
        return "Hardware"
    if "Amazon" in name:
        return "Commerce"
    if "Tesla" in name:
        return "EV"
    return "Other"

def safe_total_score(e: int, s: int, g: int) -> int:
    # 간단 평균(발표용). 필요하면 가중치로 바꿔도 됨.
    return int(round((e + s + g) / 3))

# ----------------------------
# RAG Context builder
# ----------------------------
def build_context(company: str):
    """
    - 인덱스가 없으면 RAG OFF
    - retrieve 실패해도 RAG OFF
    """
    idx_path = os.path.join("index", "faiss.index")
    meta_path = os.path.join("index", "meta.jsonl")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return [], "RAG OFF(인덱스 없음): 결제/크레딧 추가 후 `python3 ingest.py` 실행하면 근거 인용이 활성화됩니다."

    try:
        hits = retrieve(f"{company} ESG environment social governance strategy", k=10)
        ctx_lines = []
        for h in hits:
            ctx_lines.append(f"[{h['source']} p.{h['page']}] {h['text']}")
        return hits, "\n\n".join(ctx_lines)
    except Exception as e:
        return [], f"RAG OFF(검색 실패): {e}"

# ----------------------------
# Demo output
# ----------------------------
def demo_report(company: str, depth: str) -> str:
    depth_hint = {
        "간단": "요약형",
        "표준": "표준형",
        "심화": "심화형(Trade-off 포함)"
    }[depth]

    e = DEMO_SCORE.get(company, {}).get("E", 70)
    s = DEMO_SCORE.get(company, {}).get("S", 70)
    g = DEMO_SCORE.get(company, {}).get("G", 70)
    total = safe_total_score(e, s, g)

    inv, fit = DEMO_POSITION.get(company, (70, 70))

    return f"""[DEMO MODE • {depth_hint}]
Company: {company}

1) Environment
- keywords: 재생에너지, 효율, 배출관리, 공급망
- summary: (예시) 에너지 효율과 전력 조달 전략을 운영 전략과 연결합니다. Scope 관리와 공급망 배출 리스크를 관리하는 방향을 제시합니다.

2) Social
- keywords: 인재, 안전, 책임조달, 고객신뢰
- summary: (예시) 인재 확보/유지와 공급망 기준을 강화해 운영 리스크를 낮추고 신뢰를 축적합니다.

3) Governance
- keywords: 이사회감독, 컴플라이언스, 리스크관리, 투명성
- summary: (예시) 위원회 기반 감독과 내부통제 고도화를 통해 규제/평판 리스크에 대응합니다.

4) ESG Score (예시)
- E: {e}
- S: {s}
- G: {g}
- Total: {total}
- note: 데모 점수(형식 검증용)

5) ESG Risks (예시)
- E: Medium — 규제/전력비용/공급망 배출
- S: Low
- G: Medium — 규제/데이터/AI 거버넌스

6) Positioning (예시)
- investment_level: {inv}
- activity_fit: {fit}
- one-liner: ESG를 운영 효율·리스크 관리와 결합한 포지션.

7) Citations
- (데모 모드에서는 인용 없음)
"""

# ----------------------------
# OpenAI call
# ----------------------------
def call_model(prompt: str) -> str:
    if demo_mode:
        # 데모 모드는 호출하지 않음
        return ""

    if not client:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. 데모 모드를 켜거나 키/결제를 설정하세요.")

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.3
    )
    return resp.output_text

# ----------------------------
# Parse model output to extract scores/positioning (실제 모드에서도 랭킹/지도 가능)
# ----------------------------
def extract_positioning(text: str):
    if not text:
        return None
    m1 = re.search(r"investment_level\s*[:=]\s*(\d{1,3})", text, re.IGNORECASE)
    m2 = re.search(r"activity_fit\s*[:=]\s*(\d{1,3})", text, re.IGNORECASE)
    if not (m1 and m2):
        return None
    inv = max(0, min(100, int(m1.group(1))))
    fit = max(0, min(100, int(m2.group(1))))
    return inv, fit

def extract_scores(text: str):
    """
    모델 출력에서 E/S/G/Total을 최대한 유연하게 파싱.
    - "E: 78" 형태
    - "E / S / G / Total" 라인 형태 등
    실패하면 None
    """
    if not text:
        return None

    def grab(label):
        m = re.search(rf"\b{label}\b\s*[:=]\s*(\d{{1,3}})", text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    e = grab("E")
    s = grab("S")
    g = grab("G")
    total = grab("Total")

    if e is None or s is None or g is None:
        return None

    if total is None:
        total = safe_total_score(e, s, g)

    e = max(0, min(100, e))
    s = max(0, min(100, s))
    g = max(0, min(100, g))
    total = max(0, min(100, total))
    return {"E": e, "S": s, "G": g, "Total": total}

# ----------------------------
# Charts
# ----------------------------
def show_ranking_table(rows: list[dict]):
    """
    rows: [{company, group, E,S,G,Total, investment_level, activity_fit}]
    """
    if not rows:
        return

    df = pd.DataFrame(rows).copy()
    # 정렬: Total desc, tie-breaker: G desc
    df = df.sort_values(["Total", "G"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    st.markdown("## 🏆 ESG Score Ranking")
    st.caption("데모 모드에서는 고정값 기반 랭킹, 실제 모드에서는 모델 출력 파싱 기반 랭킹입니다.")
    st.dataframe(
        df[["Rank", "company", "group", "E", "S", "G", "Total"]],
        use_container_width=True,
        hide_index=True
    )

def show_positioning_map(points: list[dict]):
    """
    points: [{company, group, investment_level, activity_fit}]
    """
    if not points:
        st.info("포지셔닝 지도에 표시할 데이터가 없습니다.")
        return

    df = pd.DataFrame(points).copy()

    # 평균선 (십자선)
    avg_x = float(df["investment_level"].mean())
    avg_y = float(df["activity_fit"].mean())

    st.markdown("## 📍 ESG 전략 포지셔닝 지도")
    st.caption("X = ESG 투자 강도(investment_level) • Y = 전략 통합도(activity_fit) • 평균선을 기준으로 4분면 비교")

    base = alt.Chart(df).encode(
        x=alt.X("investment_level:Q", scale=alt.Scale(domain=[0, 100]), title="ESG 투자 강도 (0~100)"),
        y=alt.Y("activity_fit:Q", scale=alt.Scale(domain=[0, 100]), title="전략 통합도 / 활동 적합도 (0~100)"),
        color=alt.Color("group:N", legend=alt.Legend(title="Group")),
        tooltip=["company", "group", "investment_level", "activity_fit"]
    )

    points_layer = base.mark_circle(size=170, opacity=0.9)
    labels_layer = base.mark_text(align="left", dx=8, dy=-8).encode(text="company:N")

    vline = alt.Chart(pd.DataFrame({"x": [avg_x]})).mark_rule(strokeDash=[6, 6], color="gray").encode(x="x:Q")
    hline = alt.Chart(pd.DataFrame({"y": [avg_y]})).mark_rule(strokeDash=[6, 6], color="gray").encode(y="y:Q")

    # 4분면 라벨(발표용)
    quads = pd.DataFrame([
        {"label": "ESG Leaders\nHigh Invest • High Fit", "x": 82, "y": 92},
        {"label": "Heavy Spend / Low Fit\n(정렬 필요)", "x": 82, "y": 12},
        {"label": "Low Invest / High Fit\n(효율적 실행)", "x": 12, "y": 92},
        {"label": "Compliance Zone\nLow • Low", "x": 12, "y": 12},
    ])
    quad_labels = alt.Chart(quads).mark_text(
        align="left",
        opacity=0.5
    ).encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        text="label:N"
    )

    chart = (quad_labels + vline + hline + points_layer + labels_layer).interactive()
    st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Main run
# ----------------------------
if run:
    if not companies:
        st.warning("기업을 1개 이상 선택하세요.")
        st.stop()

    ranking_rows = []
    positioning_points = []

    for c in companies:
        hits, ctx = build_context(c)

        # RAG OFF면 bonus = 0
        if str(ctx).startswith("RAG OFF"):
            bonus = 0
        else:
            bonus = evidence_bonus(ctx)

        depth_rule = {
            "간단": "Be concise.",
            "표준": "Balanced detail.",
            "심화": "Be detailed; include trade-offs and uncertainty."
        }[detail]

        st.subheader(f"✅ {c}")

        # ----------------------------
        # Demo Mode
        # ----------------------------
        if demo_mode:
            st.write(demo_report(c, detail))

            e = DEMO_SCORE.get(c, {}).get("E", 70)
            s = DEMO_SCORE.get(c, {}).get("S", 70)
            g = DEMO_SCORE.get(c, {}).get("G", 70)
            total = safe_total_score(e, s, g)
            inv, fit = DEMO_POSITION.get(c, (70, 70))

            ranking_rows.append({
                "company": c,
                "group": company_group(c),
                "E": e, "S": s, "G": g, "Total": total,
                "investment_level": inv,
                "activity_fit": fit,
            })
            positioning_points.append({
                "company": c,
                "group": company_group(c),
                "investment_level": inv,
                "activity_fit": fit
            })

            with st.expander("🔎 Evidence (Top hits)"):
                st.info("데모 모드에서는 OpenAI/RAG를 호출하지 않습니다. (결제 후 ingest 실행 시 근거 인용 활성화)")
            continue

        # ----------------------------
        # Real Mode (needs API credits)
        # ----------------------------
        prompt = f"""
You are a management strategy analyst specialized in ESG.
If CONTEXT starts with "RAG OFF", do NOT invent citations. Say "인덱스 구축 전이라 인용 불가".
Company: {c}

Use ONLY the provided CONTEXT as evidence. If evidence is missing, say "insufficient evidence".
Return in Korean.

{depth_rule}

Output format:
1) Environment: 3-5 keywords + 2-3 sentence summary
2) Social: 3-5 keywords + 2-3 sentence summary
3) Governance: 3-5 keywords + 2-3 sentence summary
4) ESG Score (0-100): E / S / G / Total
   - Scoring rule: Evidence bonus factor = {bonus} (small adjustment only)
5) ESG Risks (Low/Med/High): E / S / G with reasons
6) Positioning:
   - investment_level (0-100)
   - activity_fit (0-100)
   - one-liner
7) Citations: list of "[filename p.page]" used

CONTEXT:
{ctx}

USER REQUEST:
{q}
""".strip()

        try:
            result = call_model(prompt)
            st.write(result)
        except Exception as e:
            st.error(f"모델 호출 실패: {e}")
            st.info("결제/크레딧 추가 전이라면 데모 모드를 켜서 UI 동작만 확인하세요.")
            continue

        # 모델 출력에서 점수/포지셔닝 파싱 (랭킹/지도 생성)
        scores = extract_scores(result)
        pos = extract_positioning(result)

        if scores:
            inv, fit = (None, None)
            if pos:
                inv, fit = pos

            ranking_rows.append({
                "company": c,
                "group": company_group(c),
                "E": scores["E"], "S": scores["S"], "G": scores["G"], "Total": scores["Total"],
                "investment_level": inv if inv is not None else 0,
                "activity_fit": fit if fit is not None else 0,
            })

        if pos:
            inv, fit = pos
            positioning_points.append({
                "company": c,
                "group": company_group(c),
                "investment_level": inv,
                "activity_fit": fit
            })

        with st.expander("🔎 Evidence (Top hits)"):
            if not hits:
                st.info("RAG OFF 상태입니다. 결제 후 `python3 ingest.py` 실행하면 근거가 표시됩니다.")
            else:
                for h in hits[:6]:
                    st.markdown(f"- **[{h['source']} p.{h['page']}]** (score={h['score']:.3f})")
                    st.write(h["text"][:400] + "...")

    # ----------------------------
    # Summary outputs
    # ----------------------------
    st.divider()

    if ranking_rows:
        show_ranking_table(ranking_rows)

    st.divider()

    if positioning_points:
        show_positioning_map(positioning_points)