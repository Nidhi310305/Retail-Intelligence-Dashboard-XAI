from __future__ import annotations

import html
import re
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from utils.analytics import (
    build_dashboard_summary,
    build_monthly_sales_frame,
    build_region_sales_frame,
    build_rfm_frame,
    build_sales_forecast_frame,
    detect_anomalies,
)
from utils.column_mapping import EXPECTED_ROLES, infer_column_mapping, resolve_required_roles
from utils.data_loader import load_dataset, load_sample_dataset
from utils.llm import get_gemini_client, generate_insights
from utils.rag import answer_question, build_dashboard_knowledge_base


st.set_page_config(page_title="Retail Intelligence Dashboard", page_icon="📊", layout="wide")


def inject_light_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f7f8fb 0%, #eef2f7 100%);
                color: #1f2937;
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            [data-testid="stMetric"] {
                background: white;
                border-radius: 18px;
                padding: 16px 18px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: white;
                border-radius: 18px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
                padding: 1rem;
            }
            .dashboard-heading {
                margin: 0 0 1rem 0;
                padding-top: 0.2rem;
                line-height: 1.15;
                font-size: 2rem;
                font-weight: 700;
                color: #111827;
                overflow: visible;
            }
            div.stButton > button {
                background: #e5e7eb;
                color: #111827;
                border: 1px solid #cbd5e1;
                border-radius: 999px;
                padding: 0.55rem 0.9rem;
                font-weight: 600;
            }
            div.stButton > button:hover {
                background: #dbe4f0;
                color: #111827;
                border-color: #94a3b8;
            }
            div.stButton > button:focus {
                color: #111827;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "dataset": None,
        "column_mapping_result": None,
        "resolved_mapping": None,
        "dashboard_ready": False,
        "analysis": None,
        "analysis_signature": None,
        "chat_history": [],
        "clarified_columns": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# Cache wrappers to avoid repeated LLM calls across Streamlit reruns
@st.cache_data
def _cached_infer_mapping(dataset_signature: str, columns: tuple[str, ...]) -> dict:
    return infer_column_mapping(list(columns))


@st.cache_data(ttl=3600)
def _cached_generate_insights(dataset_signature: str, context: str, question: str) -> str:
    return generate_insights(context, question)


@st.cache_data(ttl=3600)
def _cached_answer_question(dataset_signature: str, query: str, kb_text: str) -> str:
    # kb_text is the joined knowledge base used as cache key
    from utils.rag import answer_question

    kb_list = [line for line in kb_text.split("\n") if line.strip()]
    return answer_question(query, kb_list)


def _reset_dataset_state() -> None:
    st.session_state.dataset = None
    st.session_state.column_mapping_result = None
    st.session_state.resolved_mapping = None
    st.session_state.dashboard_ready = False
    st.session_state.analysis = None
    st.session_state.analysis_signature = None
    st.session_state.chat_history = []
    st.session_state.clarified_columns = {}


def show_landing() -> None:
    st.title("Retail Intelligence Dashboard with Explainable AI")
    st.write("Upload your store's sales data and get instant insights.")
    st.caption("For the best results, make sure your file includes order date, product, sales, discount, and profit information.")

    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    with sample_col:
        if st.button("Try with sample data"):
            with st.spinner("Detecting your file's columns..."):
                _reset_dataset_state()
                st.session_state.dataset = load_sample_dataset(Path(__file__).parent / "Sample - Superstore.csv")
                sig = _dataset_signature(st.session_state.dataset)
                st.session_state.column_mapping_result = _cached_infer_mapping(sig, tuple(st.session_state.dataset.columns.tolist()))
            st.rerun()

    if uploaded is not None:
        with st.spinner("Detecting your file's columns..."):
            _reset_dataset_state()
            st.session_state.dataset = load_dataset(uploaded)
            sig = _dataset_signature(st.session_state.dataset)
            st.session_state.column_mapping_result = _cached_infer_mapping(sig, tuple(st.session_state.dataset.columns.tolist()))
        st.rerun()


def render_mapping_step() -> None:
    dataset = st.session_state.dataset
    mapping_result = st.session_state.column_mapping_result
    if dataset is None or mapping_result is None:
        return

    st.subheader("Column Mapping")
    st.write("Detected roles are shown below. Any uncertain required role must be confirmed before the dashboard opens.")

    mapping = mapping_result["mapping"]
    rows = []
    for role, info in mapping.items():
        rows.append(
            {
                "Role": role,
                "Expected": EXPECTED_ROLES[role],
                "Detected Column": info.get("column") or "Not resolved",
                "Confidence": round(float(info.get("confidence", 0.0)), 2),
                "Source": info.get("source", "unknown"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    unresolved_roles = mapping_result.get("unresolved_roles", [])
    if unresolved_roles:
        st.warning("Some required roles are still uncertain. Please confirm them before continuing.")
        with st.form("clarify_roles"):
            clarified = {}
            for role in unresolved_roles:
                options = ["Not sure / leave unresolved"] + list(dataset.columns)
                current_guess = mapping.get(role, {}).get("column")
                default_index = options.index(current_guess) if current_guess in options else 0
                clarified[role] = st.selectbox(
                    f"What column should be used for {role}?",
                    options,
                    index=default_index,
                    help=EXPECTED_ROLES[role],
                    key=f"clarify_{role}",
                )
            submitted = st.form_submit_button("Confirm mapping")
            if submitted:
                for role, choice in clarified.items():
                    if choice != "Not sure / leave unresolved":
                        mapping[role] = {"column": choice, "confidence": 1.0, "source": "user"}
                st.session_state.clarified_columns = clarified
                st.session_state.resolved_mapping = resolve_required_roles({"mapping": mapping})
                missing = [role for role in EXPECTED_ROLES if role not in st.session_state.resolved_mapping]
                if missing:
                    st.error(f"Please resolve these required roles before continuing: {', '.join(missing)}")
                else:
                    st.session_state.dashboard_ready = True
                    st.rerun()
    else:
        st.success("Required roles resolved. Opening dashboard.")
        st.session_state.resolved_mapping = resolve_required_roles(mapping_result)
        st.session_state.dashboard_ready = True
        st.rerun()


def _get_col(mapping: dict[str, str], role: str) -> str | None:
    return mapping.get(role)


def _escape_dollar_signs(text: str) -> str:
    return text.replace("$", "&#36;")


def _format_llm_html(text: str) -> str:
    protected = (text or "").replace("&#36;", "__DOLLAR_ENTITY__")
    escaped = html.escape(protected).replace("__DOLLAR_ENTITY__", "&#36;")
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"&#36;(\d[\d,]*(?:\.\d+)?)", r"&#36;<strong>\1</strong>", escaped)
    escaped = re.sub(r"(\d+(?:\.\d+)?)%", r"<strong>\1%</strong>", escaped)
    escaped = re.sub(r"R²\s*([0-9]+(?:\.[0-9]+)?)", r"R² <strong>\1</strong>", escaped)
    paragraphs = [paragraph.strip().replace("\n", "<br>") for paragraph in re.split(r"\n\s*\n", escaped) if paragraph.strip()]
    if not paragraphs:
        return ""
    return "".join(f'<p style="margin:0 0 0.85rem 0; line-height:1.6;">{paragraph}</p>' for paragraph in paragraphs)


def _render_llm_response(text: str) -> None:
    st.markdown(_format_llm_html(text), unsafe_allow_html=True)


def _dataset_signature(dataset: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(dataset, index=True).sum())


def _prepare_dashboard_analysis(dataset: pd.DataFrame, mapping: dict[str, str]) -> dict[str, object]:
    total_sales = float(dataset[_get_col(mapping, "sales")].sum()) if _get_col(mapping, "sales") in dataset.columns else None
    total_profit = float(dataset[_get_col(mapping, "profit")].sum()) if _get_col(mapping, "profit") in dataset.columns else None
    order_count = len(dataset)
    unique_customers = dataset[_get_col(mapping, "customer_id")].nunique() if _get_col(mapping, "customer_id") in dataset.columns else None

    monthly = build_monthly_sales_frame(dataset, _get_col(mapping, "order_date"), _get_col(mapping, "sales")) if _get_col(mapping, "order_date") and _get_col(mapping, "sales") else None
    region_sales = build_region_sales_frame(dataset, _get_col(mapping, "region"), _get_col(mapping, "sales")) if _get_col(mapping, "region") and _get_col(mapping, "sales") else None

    rfm_artifacts = None
    if _get_col(mapping, "customer_id") and _get_col(mapping, "order_date") and _get_col(mapping, "sales"):
        order_id_col = "Order ID" if "Order ID" in dataset.columns else None
        rfm_artifacts = build_rfm_frame(dataset, _get_col(mapping, "customer_id"), _get_col(mapping, "order_date"), _get_col(mapping, "sales"), order_id_col)

    anomaly_artifacts = None
    if all(_get_col(mapping, role) for role in ["sales", "quantity", "discount", "profit"]):
        anomaly_artifacts = detect_anomalies(
            dataset,
            _get_col(mapping, "sales"),
            _get_col(mapping, "quantity"),
            _get_col(mapping, "discount"),
            _get_col(mapping, "profit"),
        )

    forecast_artifacts = None
    if _get_col(mapping, "order_date") and _get_col(mapping, "sales"):
        forecast_categorical_columns = []
        for role in ["region", "category"]:
            mapped_column = _get_col(mapping, role)
            if mapped_column:
                forecast_categorical_columns.append(mapped_column)

        excluded_columns = {
            _get_col(mapping, "order_date"),
            _get_col(mapping, "sales"),
            _get_col(mapping, "profit"),
            _get_col(mapping, "discount"),
            _get_col(mapping, "quantity"),
            _get_col(mapping, "customer_id"),
            "Order ID" if "Order ID" in dataset.columns else None,
            "Product ID" if "Product ID" in dataset.columns else None,
        }
        excluded_columns = {column for column in excluded_columns if column}

        for column in dataset.columns:
            if column in excluded_columns or column in forecast_categorical_columns:
                continue
            if pd.api.types.is_numeric_dtype(dataset[column]):
                continue
            if dataset[column].nunique(dropna=True) <= 20:
                forecast_categorical_columns.append(column)

        forecast_artifacts = build_sales_forecast_frame(
            dataset,
            _get_col(mapping, "order_date"),
            _get_col(mapping, "sales"),
            _get_col(mapping, "region"),
            _get_col(mapping, "category"),
            categorical_columns=forecast_categorical_columns,
        )

    summary = build_dashboard_summary(
        total_sales=total_sales,
        total_profit=total_profit,
        anomaly_count=anomaly_artifacts.count if anomaly_artifacts is not None else None,
        anomaly_pct=anomaly_artifacts.pct if anomaly_artifacts is not None else None,
        rfm_summary=rfm_artifacts.summary if rfm_artifacts else None,
        forecast=forecast_artifacts,
        anomalies=anomaly_artifacts,
    )

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "order_count": order_count,
        "unique_customers": unique_customers,
        "monthly": monthly,
        "region_sales": region_sales,
        "rfm_artifacts": rfm_artifacts,
        "anomaly_artifacts": anomaly_artifacts,
        "forecast_artifacts": forecast_artifacts,
        "summary": summary,
    }


def render_dashboard() -> None:
    dataset = st.session_state.dataset
    mapping = st.session_state.resolved_mapping or {}
    if dataset is None or not mapping:
        return

    dataset_signature = _dataset_signature(dataset)
    if st.session_state.analysis is None or st.session_state.analysis_signature != dataset_signature:
        with st.spinner("Preparing the dashboard — this may take a moment, please be patient..."):
            st.session_state.analysis = _prepare_dashboard_analysis(dataset, mapping)
            st.session_state.analysis_signature = dataset_signature

    analysis = st.session_state.analysis or {}
    total_sales = analysis.get("total_sales")
    total_profit = analysis.get("total_profit")
    order_count = analysis.get("order_count")
    unique_customers = analysis.get("unique_customers")
    monthly = analysis.get("monthly")
    region_sales = analysis.get("region_sales")
    rfm_artifacts = analysis.get("rfm_artifacts")
    anomaly_artifacts = analysis.get("anomaly_artifacts")
    forecast_artifacts = analysis.get("forecast_artifacts")
    summary = analysis.get("summary") or {}
    anomaly_count = anomaly_artifacts.count if anomaly_artifacts is not None else None
    anomaly_pct = anomaly_artifacts.pct if anomaly_artifacts is not None else None

    st.markdown('<h2 class="dashboard-heading">Retail Insights Dashboard</h2>', unsafe_allow_html=True)

    top_metrics = st.columns(4)
    metrics = [
        ("Sales", total_sales, "#15803d"),
        ("Profit", total_profit, "#b91c1c"),
        ("Orders", order_count, "#1d4ed8"),
        ("Customers", unique_customers, "#7c3aed"),
    ]
    for column, (label, value, color) in zip(top_metrics, metrics):
        with column:
            st.metric(label, f"{value:,.0f}" if isinstance(value, (int, float)) and value is not None else "N/A")

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("### What's Working")
        working_points = []
        if summary.get("total_profit") is not None and summary["total_profit"] > 0:
            working_points.append("Profit is positive overall.")
        if rfm_artifacts is not None:
            working_points.append(f"RFM clustering identified {rfm_artifacts.summary.get('best_k', 4)} customer groups.")
        if forecast_artifacts is not None and forecast_artifacts.metrics.get("r2") is not None:
            working_points.append(f"Forecast model R² is {forecast_artifacts.metrics['r2']:.2f}; top driver appears to be {forecast_artifacts.top_feature or 'a temporal feature'}.")
        if not working_points:
            working_points.append("The dataset supports basic operational analysis, but not every advanced module is available.")
        for item in working_points:
            st.write(f"- {item}")
    with right_col:
        st.markdown("### Needs Attention")
        attention_points = []
        if anomaly_artifacts is not None and anomaly_artifacts.count > 0:
            attention_points.append(f"{anomaly_artifacts.count} transactions ({anomaly_artifacts.pct}%) were flagged as unusual.")
        if forecast_artifacts is not None and forecast_artifacts.metrics.get("r2", 0) < 0.2:
            attention_points.append("The sales forecast is weak and should be treated as directional, not production-grade.")
        if total_profit is not None and total_profit < 0:
            attention_points.append("Profit is negative overall.")
        if not attention_points:
            attention_points.append("No obvious risk flag surfaced from the available modules.")
        for item in attention_points:
            st.write(f"- {item}")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.markdown("### Sales Trend")
        if monthly is not None and not monthly.empty:
            fig = px.line(monthly, x=_get_col(mapping, "order_date"), y="Sales", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monthly trend skipped because the dataset is missing a usable date or sales column.")
    with chart_col_2:
        st.markdown("### Sales by Region")
        if region_sales is not None and not region_sales.empty:
            fig = px.bar(region_sales, x=_get_col(mapping, "region"), y=_get_col(mapping, "sales"), color=_get_col(mapping, "region"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region chart skipped because the dataset is missing region or sales.")

    lower_col_1, lower_col_2 = st.columns(2)
    with lower_col_1:
        st.markdown("### RFM Segmentation")
        if rfm_artifacts is not None and not rfm_artifacts.frame.empty:
            fig = px.scatter_3d(
                rfm_artifacts.frame,
                x="Recency",
                y="Frequency",
                z="Monetary",
                color="Segment",
                hover_data=["Customer"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("RFM segmentation skipped because the dataset does not expose enough customer/order detail.")
    with lower_col_2:
        st.markdown("### Anomaly Detection")
        if anomaly_artifacts is not None:
            anomaly_df = anomaly_artifacts.frame.dropna(subset=["Anomaly"])
            anomaly_df = anomaly_df.assign(Flag=anomaly_df["Anomaly"].map({1.0: "Normal", -1.0: "Anomaly"}))
            fig = px.scatter(
                anomaly_df,
                x=_get_col(mapping, "discount"),
                y=_get_col(mapping, "profit"),
                color="Flag",
                hover_data=[_get_col(mapping, "sales"), _get_col(mapping, "quantity")],
                color_discrete_map={"Normal": "#9cc9ff", "Anomaly": "#ef4444"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Anomaly detection skipped because the dataset is missing sales, quantity, discount, or profit.")

    st.markdown("### Forecasting and SHAP")
    if forecast_artifacts is not None:
        st.write(f"Top factor driving the forecast: **{forecast_artifacts.top_feature or 'unknown'}**")
        st.write(
            f"Model quality: MAE {forecast_artifacts.metrics['mae']:.2f}, RMSE {forecast_artifacts.metrics['rmse']:.2f}, R² {forecast_artifacts.metrics['r2']:.2f}."
        )
        show_technical = st.toggle("Show technical SHAP view", value=False)
        if show_technical and forecast_artifacts.shap_values is not None:
            importance = abs(forecast_artifacts.shap_values).mean(axis=0)
            shap_frame = pd.DataFrame({"Feature": forecast_artifacts.feature_names, "Mean |SHAP|": importance})
            fig = px.bar(shap_frame.sort_values("Mean |SHAP|", ascending=False), x="Feature", y="Mean |SHAP|")
            st.plotly_chart(fig, use_container_width=True)
        elif show_technical:
            st.info("SHAP technical view is unavailable for this dataset or model run.")
    else:
        st.info("Forecasting skipped because the dataset is too small or lacks required temporal/sales structure.")

    st.markdown("### Business Insights")
    client = get_gemini_client()
    if client is None:
        st.warning("Insights are temporarily unavailable.")
    else:
        knowledge_base = build_dashboard_knowledge_base(
            summary_chunks=[
                f"Total sales: {total_sales:,.0f}" if total_sales is not None else "Total sales not available.",
                f"Total profit: {total_profit:,.0f}" if total_profit is not None else "Total profit not available.",
            ],
            region_sales=region_sales,
            monthly_sales=monthly,
            rfm_frame=rfm_artifacts.frame if rfm_artifacts is not None else None,
            rfm_summary=rfm_artifacts.summary if rfm_artifacts is not None else None,
            anomaly_artifacts=anomaly_artifacts,
            forecast_artifacts=forecast_artifacts,
        )
        llm_question = "Write two concise paragraphs: one on what is working and one on what needs attention. Use bold for key figures, currency values, and percentages."
        with st.spinner("Preparing your answer — this can take a minute, please be patient..."):
            kb_text = "\n".join(knowledge_base)
            llm_answer = _cached_generate_insights(dataset_signature, kb_text, llm_question)
            _render_llm_response(_escape_dollar_signs(llm_answer))

    st.markdown("### Chat With the Dashboard")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about the findings, anomalies, sales trend, or customer segments")
    if query:
        kb = build_dashboard_knowledge_base(
            summary_chunks=[f"Summary: {summary}"],
            region_sales=region_sales,
            monthly_sales=monthly,
            rfm_frame=rfm_artifacts.frame if rfm_artifacts is not None else None,
            rfm_summary=rfm_artifacts.summary if rfm_artifacts is not None else None,
            anomaly_artifacts=anomaly_artifacts,
            forecast_artifacts=forecast_artifacts,
        )
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("Preparing your answer — this can take a minute, please be patient..."):
                kb_text = "\n".join(kb)
                answer = _cached_answer_question(dataset_signature, query, kb_text)
                _render_llm_response(_escape_dollar_signs(answer))
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


def main() -> None:
    inject_light_theme()
    init_state()

    if st.session_state.dataset is None:
        show_landing()
        return

    if not st.session_state.dashboard_ready:
        render_mapping_step()
        return

    render_dashboard()


if __name__ == "__main__":
    main()
