from __future__ import annotations

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
        "chat_history": [],
        "clarified_columns": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def show_landing() -> None:
    st.title("Retail Intelligence Dashboard with Explainable AI")
    st.write("Upload your store's sales data and get instant insights.")

    upload_col, sample_col = st.columns([3, 1])
    with upload_col:
        uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    with sample_col:
        if st.button("Try with sample data"):
            st.session_state.dataset = load_sample_dataset(Path(__file__).parent / "Sample - Superstore.csv")
            st.session_state.column_mapping_result = infer_column_mapping(st.session_state.dataset.columns.tolist())
            st.session_state.dashboard_ready = False
            st.rerun()

    if uploaded is not None:
        st.session_state.dataset = load_dataset(uploaded)
        st.session_state.column_mapping_result = infer_column_mapping(st.session_state.dataset.columns.tolist())
        st.session_state.dashboard_ready = False
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


def render_dashboard() -> None:
    dataset = st.session_state.dataset
    mapping = st.session_state.resolved_mapping or {}
    if dataset is None or not mapping:
        return

    st.subheader("Adaptive Dashboard")

    total_sales = float(dataset[_get_col(mapping, "sales")].sum()) if _get_col(mapping, "sales") in dataset.columns else None
    total_profit = float(dataset[_get_col(mapping, "profit")].sum()) if _get_col(mapping, "profit") in dataset.columns else None
    order_count = len(dataset)
    unique_customers = dataset[_get_col(mapping, "customer_id")].nunique() if _get_col(mapping, "customer_id") in dataset.columns else None
    anomaly_count = None
    anomaly_pct = None

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

    if _get_col(mapping, "order_date") and _get_col(mapping, "sales"):
        monthly = build_monthly_sales_frame(dataset, _get_col(mapping, "order_date"), _get_col(mapping, "sales"))
    else:
        monthly = None

    region_sales = None
    if _get_col(mapping, "region") and _get_col(mapping, "sales"):
        region_sales = build_region_sales_frame(dataset, _get_col(mapping, "region"), _get_col(mapping, "sales"))

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
        if anomaly_artifacts is not None:
            anomaly_count = anomaly_artifacts.count
            anomaly_pct = anomaly_artifacts.pct

    forecast_artifacts = None
    if _get_col(mapping, "order_date") and _get_col(mapping, "sales"):
        forecast_artifacts = build_sales_forecast_frame(
            dataset,
            _get_col(mapping, "order_date"),
            _get_col(mapping, "sales"),
            _get_col(mapping, "region"),
            _get_col(mapping, "category"),
        )

    summary = build_dashboard_summary(
        total_sales=total_sales,
        total_profit=total_profit,
        anomaly_count=anomaly_count,
        anomaly_pct=anomaly_pct,
        rfm_summary=rfm_artifacts.summary if rfm_artifacts else None,
        forecast=forecast_artifacts,
        anomalies=anomaly_artifacts,
    )

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
        llm_question = "Write two concise paragraphs: one on what is working and one on what needs attention."
        with st.spinner("Preparing your answer..."):
            llm_answer = generate_insights("\n".join(knowledge_base), llm_question)
            st.write(llm_answer)

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
            with st.spinner("Preparing your answer..."):
                answer = answer_question(query, kb)
                st.write(answer)
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
