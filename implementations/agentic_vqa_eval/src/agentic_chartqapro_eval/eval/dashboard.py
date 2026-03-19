"""Interactive Streamlit dashboard for browsing MEPs and evaluation results.

Launch with:
    streamlit run src/agentic_chartqapro_eval/eval/dashboard.py

Or via the module alias added in setup.py:
    ./.venv/bin/streamlit run src/agentic_chartqapro_eval/eval/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd


try:
    import streamlit as st
except ImportError:
    raise SystemExit("Install streamlit first:  pip install streamlit matplotlib") from None

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image  # noqa: F401

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ChartQAPro Eval Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_metrics(path: str) -> pd.DataFrame:
    """Load a metrics JSONL file and return it as a DataFrame."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


@st.cache_data
def load_taxonomy(path: str) -> pd.DataFrame:
    """Load a taxonomy JSONL file and return it as a DataFrame."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


@st.cache_data
def load_meps(mep_dir: str) -> dict:
    """Load all MEP JSON files from directory. Returns {sample_id: mep_dict}."""
    meps = {}
    for p in sorted(Path(mep_dir).glob("*.json")):
        try:
            m = json.loads(p.read_text())
            sid = m.get("sample", {}).get("sample_id", p.stem)
            meps[sid] = m
        except Exception:
            pass
    return meps


# ---------------------------------------------------------------------------
# Sidebar: configuration  # noqa: ERA001
# ---------------------------------------------------------------------------

st.sidebar.title("📊 ChartQAPro Eval")
st.sidebar.markdown("---")
st.sidebar.subheader("Data Paths")

mep_dir_input = st.sidebar.text_input(
    "MEP directory",
    value="meps/gemini_gemini/chartqapro/test",
    help="Directory containing .json MEP files",
)
metrics_input = st.sidebar.text_input("metrics.jsonl", value="output/metrics.jsonl", help="Output of eval_outputs.py")
taxonomy_input = st.sidebar.text_input(
    "taxonomy.jsonl (optional)",
    value="output/taxonomy.jsonl",
    help="Output of error_taxonomy.py",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Load data
df_metrics, df_tax, meps = None, None, {}
load_errors = []

if Path(metrics_input).exists():
    df_metrics = load_metrics(metrics_input)
else:
    load_errors.append(f"metrics file not found: `{metrics_input}`")

if Path(taxonomy_input).exists():
    df_tax = load_taxonomy(taxonomy_input)

if Path(mep_dir_input).is_dir():
    meps = load_meps(mep_dir_input)
else:
    load_errors.append(f"MEP directory not found: `{mep_dir_input}`")

# Filters (applied after load)
question_types = []
if df_metrics is not None and "question_type" in df_metrics.columns:
    question_types = sorted(df_metrics["question_type"].unique().tolist())

selected_types = st.sidebar.multiselect(
    "Question types",
    options=question_types,
    default=question_types,
    help="Filter by question type",
)

verdict_options = ["confirmed", "revised", "skipped"]
selected_verdicts = st.sidebar.multiselect(
    "Verifier verdict",
    options=verdict_options,
    default=verdict_options,
    help="Filter by VerifierAgent verdict",
)

failure_types = []
if df_tax is not None and "failure_type" in df_tax.columns:
    failure_types = sorted(df_tax["failure_type"].unique().tolist())
selected_failures = st.sidebar.multiselect("Failure types (taxonomy)", options=failure_types, default=failure_types)

st.sidebar.markdown("---")
st.sidebar.caption("agentic_chartqapro_eval · ChartQAPro Dashboard")


# ---------------------------------------------------------------------------
# Main content: two tabs
# ---------------------------------------------------------------------------

tab_overview, tab_browser = st.tabs(["📈 Overview", "🔍 Sample Browser"])


# ── Tab 1: Overview ──────────────────────────────────────────────────────────

with tab_overview:
    if load_errors:
        for err in load_errors:
            st.error(err)
        st.info("Adjust the paths in the sidebar and make sure you've run the pipeline first.")
        st.stop()

    # Apply filters
    df = df_metrics.copy()
    if selected_types:
        df = df[df["question_type"].isin(selected_types)]
    if "verifier_verdict" in df.columns and selected_verdicts:
        df = df[df["verifier_verdict"].isin(selected_verdicts)]

    if len(df) == 0:
        st.warning("No rows match the current filters.")
        st.stop()

    # ── Summary metrics ───────────────────────────────────────────────────
    st.subheader("Summary")
    n = len(df)
    avg_acc = df["answer_accuracy"].mean()
    avg_lat = df["latency_sec"].mean() if "latency_sec" in df.columns else 0.0
    correct = int((df["answer_accuracy"] >= 1.0).sum())

    has_verifier = "verifier_verdict" in df.columns and not df["verifier_verdict"].eq("skipped").all()
    revised_n = int((df["verifier_verdict"] == "revised").sum()) if has_verifier else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Samples", n)
    c2.metric("Accuracy", f"{avg_acc:.1%}", f"{correct} correct")
    c3.metric("Avg Latency", f"{avg_lat:.1f}s")
    c4.metric("Correct", correct, f"{correct / n:.1%}")
    c5.metric("Verifier Revised", revised_n, f"{revised_n / n:.1%}" if n else "—")

    st.markdown("---")

    # ── Accuracy by question type ─────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Accuracy by Question Type")
        if HAS_MPL:
            by_type = df.groupby("question_type")["answer_accuracy"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ["#d63031" if v < 0.4 else "#fdcb6e" if v < 0.75 else "#00b894" for v in by_type.values]
            ax.barh(by_type.index, by_type.values, color=colors, edgecolor="white")
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Avg Accuracy")
            ax.axvline(avg_acc, color="#6c5ce7", linestyle="--", linewidth=1.2)
            for i, (_qt, val) in enumerate(by_type.items()):
                ax.text(val + 0.01, i, f"{val:.1%}", va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.dataframe(df.groupby("question_type")["answer_accuracy"].mean().round(3))

    # ── Verifier stats ────────────────────────────────────────────────────
    with col_b:
        st.subheader("Verifier Verdict Distribution")
        if has_verifier and HAS_MPL:
            v_counts = df["verifier_verdict"].value_counts()
            vc_colors = {
                "confirmed": "#00b894",
                "revised": "#fdcb6e",
                "skipped": "#b2bec3",
            }
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(
                v_counts.index,
                v_counts.values,
                color=[vc_colors.get(k, "#6c5ce7") for k in v_counts.index],
                edgecolor="white",
                width=0.5,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(int(bar.get_height())),
                    ha="center",
                    fontsize=10,
                )
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        elif not has_verifier:
            st.info("Verifier was not used (all verdicts are 'skipped').")
        else:
            st.dataframe(df["verifier_verdict"].value_counts())

    # ── Failure taxonomy ──────────────────────────────────────────────────
    if df_tax is not None:
        st.markdown("---")
        st.subheader("Failure Taxonomy (Pass 4)")

        tax = df_tax.copy()
        if selected_failures:
            tax = tax[tax["failure_type"].isin(selected_failures)]

        col_c, col_d = st.columns([2, 1])
        with col_c:
            if HAS_MPL:
                counts = tax["failure_type"].value_counts()
                palette = {
                    "correct": "#00b894",
                    "extraction_error": "#e84393",
                    "axis_misread": "#e17055",
                    "arithmetic_mistake": "#fdcb6e",
                    "legend_confusion": "#fd79a8",
                    "hallucinated_element": "#d63031",
                    "unanswerable_failure": "#0984e3",
                    "question_misunderstanding": "#6c5ce7",
                    "other": "#b2bec3",
                }
                fig, ax = plt.subplots(figsize=(7, 4))
                bar_colors = [palette.get(k, "#6c5ce7") for k in counts.index[::-1]]
                ax.barh(
                    counts.index[::-1],
                    counts.values[::-1],
                    color=bar_colors,
                    edgecolor="white",
                )
                for i, (_ft, c) in enumerate(zip(counts.index[::-1], counts.values[::-1])):
                    ax.text(c + 0.1, i, str(c), va="center", fontsize=9)
                ax.set_xlabel("Count")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.dataframe(tax["failure_type"].value_counts())

        with col_d:
            st.dataframe(
                tax["failure_type"].value_counts().rename("count").reset_index(),
                use_container_width=True,
                height=300,
            )

    # ── Judge scores ──────────────────────────────────────────────────────
    judge_cols = [c for c in df.columns if c.startswith("judge_") and df[c].dtype in ["float64", "int64"]]
    if judge_cols:
        st.markdown("---")
        st.subheader("LLM Judge Rubric Scores")
        judge_means = df[judge_cols].mean().rename(lambda c: c.replace("judge_", "").replace("_", " ").title())
        st.dataframe(judge_means.round(3).to_frame("Mean Score"), use_container_width=False)

        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(
                judge_means.index,
                judge_means.values,
                color="#6c5ce7",
                edgecolor="white",
                alpha=0.85,
            )
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
            ax.set_title("Judge Scores (mean)")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ── Tab 2: Sample Browser ────────────────────────────────────────────────────

with tab_browser:
    st.subheader("Sample Browser")

    if not meps:
        st.error(f"No MEPs loaded from `{mep_dir_input}`. Check the path in the sidebar.")
        st.stop()

    # Merge metrics into browser if available
    sample_ids = list(meps.keys())
    metrics_by_id = df_metrics.set_index("sample_id").to_dict("index") if df_metrics is not None else {}

    tax_by_id = (
        (df_tax.set_index("sample_id").to_dict("index") if "sample_id" in df_tax.columns else {})
        if df_tax is not None
        else {}
    )

    # Filter sample IDs based on sidebar filter
    if selected_types and df_metrics is not None:
        valid_ids = set(df_metrics[df_metrics["question_type"].isin(selected_types)]["sample_id"].tolist())
        sample_ids = [s for s in sample_ids if s in valid_ids]

    if not sample_ids:
        st.warning("No samples match the current filters.")
        st.stop()

    selected_id = st.selectbox("Select sample", sample_ids)
    mep = meps[selected_id]

    sample = mep.get("sample", {})
    plan = mep.get("plan", {}).get("parsed", {})
    vision = mep.get("vision", {}).get("parsed", {})
    verifier = (mep.get("verifier") or {}).get("parsed", {})
    timestamps = mep.get("timestamps", {})
    errors = mep.get("errors", [])

    m_row = metrics_by_id.get(selected_id, {})
    t_row = tax_by_id.get(selected_id, {})

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # Chart image
        img_path = sample.get("image_ref", {}).get("path", "")
        if img_path and Path(img_path).exists() and HAS_PIL:
            st.image(img_path, caption=f"Chart: {selected_id}", use_container_width=True)
        elif img_path:
            st.warning(f"Image not found: {img_path}")
        else:
            st.info("No image path in MEP.")

    with col_right:
        # Sample metadata
        st.markdown(f"**Question:** {sample.get('question', '—')}")
        st.markdown(
            f"**Type:** `{sample.get('question_type', '—')}` &nbsp;|&nbsp; **Expected:** `{sample.get('expected_output', '—')}`"
        )

        # Accuracy badge
        acc = m_row.get("answer_accuracy", None)
        if acc is not None:
            color = "green" if acc >= 1.0 else "orange" if acc >= 0.5 else "red"
            st.markdown(f"**Accuracy:** :{color}[{acc:.2f}]")

        st.markdown("---")

        # Planner steps
        steps = plan.get("steps", [])
        with st.expander(f"Planner steps ({len(steps)})", expanded=True):
            for i, s in enumerate(steps, 1):
                st.markdown(f"{i}. {s}")

        # Vision agent
        with st.expander("VisionAgent output", expanded=True):
            st.markdown(f"**Draft answer:** `{vision.get('answer', '—')}`")
            st.markdown(f"**Explanation:** {vision.get('explanation', '—')}")

        # Verifier
        if verifier:
            verdict = verifier.get("verdict", "—")
            v_color = "green" if verdict == "confirmed" else "orange" if verdict == "revised" else "gray"
            with st.expander(f"VerifierAgent — :{v_color}[{verdict}]", expanded=True):
                st.markdown(f"**Final answer:** `{verifier.get('answer', '—')}`")
                st.markdown(f"**Reasoning:** {verifier.get('reasoning', '—')}")

        # Failure taxonomy
        if t_row:
            ft = t_row.get("failure_type", "—")
            ft_color = "green" if ft == "correct" else "red"
            with st.expander(f"Failure taxonomy — :{ft_color}[{ft}]"):
                st.markdown(f"**Reason:** {t_row.get('failure_reason', '—')}")

        # Latency
        if timestamps:
            total_ms = sum(
                [
                    timestamps.get("planner_ms", 0),
                    timestamps.get("vision_ms", 0),
                    timestamps.get("verifier_ms", 0),
                ]
            )
            st.caption(
                f"Latency — planner: {timestamps.get('planner_ms', 0):.0f}ms  "
                f"vision: {timestamps.get('vision_ms', 0):.0f}ms  "
                f"verifier: {timestamps.get('verifier_ms', 0):.0f}ms  "
                f"**total: {total_ms / 1000:.2f}s**"
            )

        # Errors
        if errors:
            with st.expander(f"⚠ Errors ({len(errors)})"):
                for e in errors:
                    st.code(e)
