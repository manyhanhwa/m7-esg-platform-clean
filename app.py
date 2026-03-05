import os
import re
import streamlit as st

import pandas as pd
import altair as alt

# OpenAI는 "Real 모드"에서만 사용 (키 없으면 호출 안 하게 방어)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# rag.py가 키 없을 때도 import 단계에서 죽지 않도록 이미 수정한 전제
# 그래도 안전하게 한 번 더 방어
try:
    from rag import retrieve
except Exception:
    retrieve = None

from scoring import evidence_bonus

# =========================
# Page config
# =========================
st.set_page_config(page_title="한시서가 • M7 ESG Strategy", page_icon="📚", layout="wide")

# =========================
# Han-si-seoga SaaS UI Theme (warm minimal)
# =========================
st.markdown(
    """
<style>
/* 전체 배경 */
.stApp {
  background: radial-gradient(900px circle at 15% 12%, rgba(227, 203, 174, 0.28), transparent 45%),
              radial-gradient(750px circle at 88% 18%, rgba(210, 230, 214, 0.22), transparent 40%),
              #F7F1E6;
  color: #2B2620;
}

/* 본문 폭/여백 */
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1200px; }

/* 기본 타이포 */
html, body, [class*="css"] {
  font-family: ui-serif, Georgia, "Apple SD Gothic Neo", "Noto Sans KR", system-ui, -apple-system, sans-serif;
}

/* 헤더 계열 */
h1, h2, h3 { color: #2B2620; letter-spacing: -0.02em; }
a { color: #2B2620; }

/* 카드 */
.card {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(72, 62, 52, 0.14);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 26px rgba(20, 16, 12, 0.08);
}

/* 약한 텍스트 */
.muted { color: rgba(43, 38, 32, 0.65); }

/* KPI */
.kpi-title { font-size: 12px; letter-spacing: 0.02em; }
.kpi-num { font-size: 28px; font-weight: 900; margin-top: 6px; }
.kpi-sub { font-size: 12px; margin-top: 4px; }

/* 버튼 */
div.stButton > button {
  width: 100%;
  border-radius: 14px;
  height: 46px;
  font-weight: 800;
  border: 1px solid rgba(72, 62, 52, 0.22);
  background: linear-gradient(135deg, #3E3226, #6B553C);
  color: #F7F1E6;
}
div.stButton > button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
  transition: 0.12s ease;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.55);
  border-right: 1px solid rgba(72, 62, 52, 0.12);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }

/* 탭 */
button[data-baseweb="tab"] {
  font-weight: 800 !important;
}

/* Expander */
div[data-testid="stExpander"] > details {
  background: rgba(255,255,255,0.58);
  border: 1px solid rgba(72,62,52,0.14);
  border-radius: 14px;
  padding: 8px 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Secrets / Env (safe)
# =========================
try:
    secret_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    secret_key = None

OPENAI_API_KEY = secret_key or os.getenv("OPENAI_API_KEY")

client = None
if OpenAI and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Constants
# =========================
M7 = [
    "NVIDIA (NVDA)", "Microsoft (MSFT)", "Apple (AAPL)",
    "Alphabet (GOOGL)", "Amazon (AMZN)", "Meta (META)", "Tesla (TSLA)"
]

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

# =========================
# UI: Hero
# =========================
st.markdown(
    """
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
    <div>
      <div style="font-size:34px;font-weight:950;line-height:1.12;">
        📚 한시서가 감성으로 만든<br/>M7 ESG Strategy Dashboard
      </div>
      <div class="muted" style="margin-top:6px;font-size:14px;">
        따뜻한 UI · 비교 가능한 점수/리스크/포지셔닝 · (A안: 데모 → B안: RAG 근거 기반)
      </div>
    </div>
    <div class="muted" style="text-align:right;font-size:12px;min-width:210px;">
      Web App (No install)<br/>
      Streamlit Cloud 배포
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    companies = st.multiselect("기업 선택", M7, default=["NVIDIA (NVDA)", "Microsoft (MSFT)"])
    detail = st.select_slider("분석 깊이", ["간단", "표준", "심화"], value="표준")
    demo_mode = st.checkbox("데모 모드(교수님 접속용)", value=True)

    q = st.text_area(
        "요청",
        value="각 기업의 ESG 전략을 E/S/G로 요약하고, 점수/리스크/포지셔닝까지 보여줘.",
        height=110
    )

    run = st.button("분석 실행", type="primary")

    st.markdown("---")
    st.markdown(
        '<div class="muted" style="font-size:12px;">'
        '• 데모 모드: 비용 없이 UI/흐름 확인<br/>'
        '• 실제 모드: OpenAI API + (선택) RAG 인덱스 필요<br/>'
        '</div>',
        unsafe_allow_html=True,
    )

if (not demo_mode) and (not OPENAI_API_KEY):
    st.warning("데모 모드를 끄면 OpenAI API Key가 필요합니다. (Streamlit Cloud → Settings → Secrets)")

# =========================
# Helpers
# =========================
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
    return int(round((e + s + g) / 3))

def kpi_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
<div class="card">
  <div class="muted kpi-title">{title}</div>
  <div class="kpi-num">{value}</div>
  <div class="muted kpi-sub">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# RAG Context builder
# =========================
def build_context(company: str):
    idx_path = os.path.join("index", "faiss.index")
    meta_path = os.path.join("index", "meta.jsonl")

    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return [], "RAG OFF(인덱스 없음): `python3 ingest.py` 실행 후 근거 인용 활성화"

    if retrieve is None:
        return [], "RAG OFF(rag 모듈 문제): 배포 환경에서 rag.py import 실패"

    try:
        hits = retrieve(f"{company} ESG environment social governance strategy", k=10)
        ctx_lines = []
        for h in hits:
            ctx_lines.append(f"[{h['source']} p.{h['page']}] {h['text']}")
        return hits, "\n\n".join(ctx_lines)
    except Exception as e:
        return [], f"RAG OFF(검색 실패): {e}"

# =========================
# Demo output
# =========================
def demo_report(company: str, depth: str) -> str:
    depth_hint = {"간단": "요약형", "표준": "표준형", "심화": "심화형(Trade-off 포함)"}[depth]

    e = DEMO_SCORE.get(company, {}).get("E", 70)
    s = DEMO_SCORE.get(company, {}).get("S", 70)
    g = DEMO_SCORE.get(company, {}).get("G", 70)
    total = safe_total_score(e, s, g)
    inv, fit = DEMO_POSITION.get(company, (70, 70))

    return f"""[DEMO MODE • {depth_hint}]
Company: {company}

1) Environment
- keywords: 재생에너지, 효율, 배출관리, 공급망
- summary: (예시) 에너지 효율/전력 조달 전략을 운영 전략과 연결합니다. Scope 관리와 공급망 배출 리스크를 관리하는 방향을 제시합니다.

2) Social
- keywords: 인재, 안전, 책임조달, 고객신뢰
- summary: (예시) 인재 확보/유지와 공급망 기준 강화로 운영 리스크를 낮추고 신뢰를 축적합니다.

3) Governance
- keywords: 이사회감독, 컴플라이언스, 리스크관리, 투명성
- summary: (예시) 위원회 기반 감독과 내부통제 고도화를 통해 규제/평판 리스크에 대응합니다.

4) ESG Score (예시)
- E: {e}
- S: {s}
- G: {g}
- Total: {total}

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

# =========================
# OpenAI call
# =========================
def call_model(prompt: str) -> str:
    if demo_mode:
        return ""

    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 없거나 OpenAI 라이브러리 로드 실패입니다. (데모 모드 ON 권장)")

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.3
    )
    return resp.output_text

# =========================
# Parsing (for ranking/map in real mode)
# =========================
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

    e, s, g, total = [max(0, min(100, int(x))) for x in (e, s, g, total)]
    return {"E": e, "S": s, "G": g, "Total": total}

# =========================
# Charts
# =========================
def show_ranking_table(rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows).copy()
    df = df.sort_values(["Total", "G"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    st.markdown("### 🏆 ESG Score Ranking")
    st.caption("데모 모드는 고정값 기반, 실제 모드는 모델 출력 파싱 기반입니다.")
    st.dataframe(df[["Rank", "company", "group", "E", "S", "G", "Total"]], use_container_width=True, hide_index=True)

def show_positioning_map(points: list[dict]):
    if not points:
        st.info("포지셔닝 지도에 표시할 데이터가 없습니다.")
        return

    df = pd.DataFrame(points).copy()

    avg_x = float(df["investment_level"].mean())
    avg_y = float(df["activity_fit"].mean())

    st.markdown("### 📍 ESG 전략 포지셔닝 지도")
    st.caption("X=ESG 투자 강도 • Y=전략 통합도(활동 적합도) • 평균선 기준 4분면 비교")

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

    quads = pd.DataFrame([
        {"label": "Leaders\nHigh • High", "x": 82, "y": 92},
        {"label": "Spend↑ Fit↓\n정렬 필요", "x": 82, "y": 12},
        {"label": "Spend↓ Fit↑\n효율적 실행", "x": 12, "y": 92},
        {"label": "Compliance\nLow • Low", "x": 12, "y": 12},
    ])
    quad_labels = alt.Chart(quads).mark_text(align="left", opacity=0.45).encode(
        x=alt.X("x:Q"), y=alt.Y("y:Q"), text="label:N"
    )

    chart = (quad_labels + vline + hline + points_layer + labels_layer).interactive()
    st.altair_chart(chart, use_container_width=True)

# =========================
# Main
# =========================
tab_overview, tab_ranking, tab_position, tab_details = st.tabs(
    ["📌 Overview", "🏆 Ranking", "📍 Positioning", "📄 Details"]
)

if run:
    if not companies:
        st.warning("기업을 1개 이상 선택하세요.")
        st.stop()

    ranking_rows = []
    positioning_points = []
    details_blocks = []

    # --- Analyze each company ---
    for c in companies:
        hits, ctx = build_context(c)

        bonus = 0 if str(ctx).startswith("RAG OFF") else evidence_bonus(ctx)

        depth_rule = {
            "간단": "Be concise.",
            "표준": "Balanced detail.",
            "심화": "Be detailed; include trade-offs and uncertainty."
        }[detail]

        # --- Demo mode ---
        if demo_mode:
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

            details_blocks.append((c, demo_report(c, detail), hits))
            continue

        # --- Real mode ---
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
        except Exception as e:
            result = f"모델 호출 실패: {e}\n\n(교수님 공유용이면 데모 모드를 ON으로 두세요.)"

        # collect for details tab
        details_blocks.append((c, result, hits))

        # parse for ranking/map
        scores = extract_scores(result)
        pos = extract_positioning(result)

        if scores:
            inv, fit = pos if pos else (0, 0)
            ranking_rows.append({
                "company": c,
                "group": company_group(c),
                "E": scores["E"], "S": scores["S"], "G": scores["G"], "Total": scores["Total"],
                "investment_level": inv,
                "activity_fit": fit,
            })

        if pos:
            inv, fit = pos
            positioning_points.append({
                "company": c,
                "group": company_group(c),
                "investment_level": inv,
                "activity_fit": fit
            })

    # --- Overview tab (KPI cards) ---
    with tab_overview:
        st.markdown("### 오늘의 대시보드")
        c1, c2, c3, c4 = st.columns(4)

        n = len(companies)
        avg_total = int(round(pd.DataFrame(ranking_rows)["Total"].mean())) if ranking_rows else "—"
        demo_badge = "ON" if demo_mode else "OFF"
        rag_badge = "OFF" if (not os.path.exists(os.path.join("index", "faiss.index"))) else "ON"

        with c1:
            kpi_card("선택 기업 수", str(n), "M7 중 분석 대상")
        with c2:
            kpi_card("평균 ESG Total", str(avg_total), "현재 선택 기업 평균")
        with c3:
            kpi_card("데모 모드", demo_badge, "교수님 접속용 안전 모드")
        with c4:
            kpi_card("RAG 상태", rag_badge, "인덱스 있으면 ON")

        st.markdown(
            '<div class="muted" style="margin-top:10px;">'
            'Tip) 발표 때는 “A안(데모)으로 플랫폼 구조를 검증 → B안(RAG 근거 기반)으로 고도화” 흐름으로 설명하면 점수 잘 나옵니다.'
            '</div>',
            unsafe_allow_html=True,
        )

    # --- Ranking tab ---
    with tab_ranking:
        show_ranking_table(ranking_rows)

    # --- Position tab ---
    with tab_position:
        show_positioning_map(positioning_points)

    # --- Details tab ---
    with tab_details:
        st.markdown("### 📄 기업별 결과")
        for c, text, hits in details_blocks:
            st.markdown(f"#### ✅ {c}")
            st.write(text)
            with st.expander("🔎 Evidence (Top hits)"):
                if demo_mode:
                    st.info("데모 모드에서는 근거 인용을 표시하지 않습니다.")
                else:
                    if not hits:
                        st.info("RAG OFF 상태입니다. 인덱스 구축 후 근거가 표시됩니다.")
                    else:
                        for h in hits[:6]:
                            st.markdown(f"- **[{h.get('source','?')} p.{h.get('page','?')}]** (score={h.get('score',0):.3f})")
                            st.write(str(h.get("text", ""))[:400] + "...")

else:
    with tab_overview:
        st.markdown(
            """
<div class="card">
  <div style="font-size:16px;font-weight:900;">시작 안내</div>
  <div class="muted" style="margin-top:6px;">
    왼쪽 사이드바에서 기업을 선택하고 <b>분석 실행</b>을 누르세요.<br/>
    교수님 공유용이면 <b>데모 모드 ON</b>으로 두면 설치/키 없이도 정상 동작합니다.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    '<div class="muted" style="text-align:center;font-size:12px;">'
    '© 한시서가 스타일 • M7 ESG Strategy Dashboard • (A안 데모 → B안 RAG 확장)'
    '</div>',
    unsafe_allow_html=True,
)