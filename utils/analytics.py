from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


@dataclass
class ForecastArtifacts:
    model: RandomForestRegressor
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: np.ndarray
    metrics: dict[str, float]
    shap_values: np.ndarray | None
    feature_names: list[str]
    top_feature: str | None


@dataclass
class RfmArtifacts:
    frame: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class AnomalyArtifacts:
    frame: pd.DataFrame
    count: int
    pct: float
    top_features: list[str]


def _require_columns(dataset: pd.DataFrame, required: list[str]) -> bool:
    return all(column in dataset.columns for column in required)


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def build_monthly_sales_frame(dataset: pd.DataFrame, date_column: str, sales_column: str) -> pd.DataFrame:
    if not _require_columns(dataset, [date_column, sales_column]):
        return pd.DataFrame()
    working = dataset.copy()
    working[date_column] = _safe_datetime(working[date_column])
    working = working.dropna(subset=[date_column])
    monthly = working.groupby(working[date_column].dt.to_period("M"))[sales_column].sum().reset_index()
    monthly[date_column] = monthly[date_column].astype(str)
    return monthly.rename(columns={sales_column: "Sales"})


def build_region_sales_frame(dataset: pd.DataFrame, region_column: str, sales_column: str) -> pd.DataFrame:
    if not _require_columns(dataset, [region_column, sales_column]):
        return pd.DataFrame()
    return dataset.groupby(region_column)[sales_column].sum().reset_index().sort_values(sales_column, ascending=False)


def build_rfm_frame(dataset: pd.DataFrame, customer_column: str, order_date_column: str, sales_column: str, order_id_column: str | None = None) -> RfmArtifacts | None:
    required = [customer_column, order_date_column, sales_column]
    if not _require_columns(dataset, required):
        return None

    working = dataset.copy()
    working[order_date_column] = _safe_datetime(working[order_date_column])
    working = working.dropna(subset=[customer_column, order_date_column])
    if working.empty:
        return None

    reference_date = working[order_date_column].max() + pd.Timedelta(days=1)
    aggregation: dict[str, Any] = {
        order_date_column: lambda x: (reference_date - x.max()).days,
        sales_column: "sum",
    }
    if order_id_column and order_id_column in working.columns:
        aggregation[order_id_column] = "nunique"
    else:
        working = working.copy()
        working["_row_id"] = 1
        order_id_column = "_row_id"
        aggregation[order_id_column] = "sum"

    rfm = working.groupby(customer_column).agg(aggregation).reset_index()
    count_column = order_id_column if order_id_column != "_row_id" else "_row_id"
    rfm.columns = ["Customer", "Recency", "Frequency", "Monetary"]

    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency_log", "Frequency", "Monetary_log"]])

    if len(rfm) < 4:
        rfm["Cluster"] = 0
        rfm["Segment"] = "Small Sample"
        return RfmArtifacts(frame=rfm, summary={"note": "Too few customers for stable clustering."})

    silhouette_by_k: dict[int, float] = {}
    best_k = 4
    best_score = -1.0
    for k in range(2, min(11, len(rfm))):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        silhouette_by_k[k] = float(score)
        if score > best_score:
            best_score = score
            best_k = k

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(1).to_dict("index")
    size_summary = rfm["Cluster"].value_counts().to_dict()
    rfm["Segment"] = rfm["Cluster"].map({
        0: "Champions",
        1: "Loyal High-Value",
        2: "At-Risk",
        3: "Lost/Churned",
    }).fillna("Other")

    return RfmArtifacts(
        frame=rfm,
        summary={
            "best_k": best_k,
            "silhouette_by_k": silhouette_by_k,
            "cluster_summary": cluster_summary,
            "size_summary": size_summary,
            "segment_counts": rfm["Segment"].value_counts().to_dict(),
        },
    )


def detect_anomalies(dataset: pd.DataFrame, sales_column: str, quantity_column: str, discount_column: str, profit_column: str) -> AnomalyArtifacts | None:
    if not _require_columns(dataset, [sales_column, quantity_column, discount_column, profit_column]):
        return None

    working = dataset[[sales_column, quantity_column, discount_column, profit_column]].copy().dropna()
    if working.empty:
        return None

    forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    labels = forest.fit_predict(working)
    anomaly_mask = labels == -1

    top_features = list(working.columns[np.argsort(np.abs(working.corrwith(working[profit_column])).values)[::-1][:2]])
    frame = dataset.copy()
    frame["Anomaly"] = np.nan
    frame.loc[working.index, "Anomaly"] = labels

    count = int(anomaly_mask.sum())
    pct = round(count / len(working) * 100, 1)
    return AnomalyArtifacts(frame=frame, count=count, pct=pct, top_features=top_features)


def build_sales_forecast_frame(
    dataset: pd.DataFrame,
    date_column: str,
    sales_column: str,
    region_column: str | None = None,
    category_column: str | None = None,
    categorical_columns: list[str] | None = None,
) -> ForecastArtifacts | None:
    if not _require_columns(dataset, [date_column, sales_column]):
        return None

    working = dataset.copy()
    working[date_column] = _safe_datetime(working[date_column])
    working = working.dropna(subset=[date_column, sales_column])
    if working.empty:
        return None

    working["Year"] = working[date_column].dt.year
    working["Month"] = working[date_column].dt.month
    working["Quarter"] = working[date_column].dt.quarter

    group_columns = ["Year", "Month", "Quarter"]
    candidate_columns: list[str] = []
    for column in [region_column, category_column, *(categorical_columns or [])]:
        if column and column in working.columns and column not in candidate_columns:
            candidate_columns.append(column)

    for column in working.columns:
        if column in {date_column, sales_column, "Year", "Month", "Quarter"}:
            continue
        if pd.api.types.is_numeric_dtype(working[column]):
            continue
        if column not in candidate_columns and working[column].nunique(dropna=True) <= 20:
            candidate_columns.append(column)

    group_columns.extend(candidate_columns)

    monthly = working.groupby(group_columns)[sales_column].sum().reset_index()
    if len(monthly) < 8:
        return None

    feature_frame = monthly.drop(columns=[sales_column]).copy()
    for column in feature_frame.columns:
        if not pd.api.types.is_numeric_dtype(feature_frame[column]):
            encoder = LabelEncoder()
            feature_frame[column] = encoder.fit_transform(feature_frame[column].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(feature_frame, monthly[sales_column], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    shap_values = None
    top_feature = None
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        importance = np.abs(shap_values).mean(axis=0)
        top_feature = str(X_test.columns[int(np.argmax(importance))])
    except Exception:
        shap_values = None

    return ForecastArtifacts(
        model=model,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        metrics=metrics,
        shap_values=shap_values,
        feature_names=list(X_test.columns),
        top_feature=top_feature,
    )


def build_dashboard_summary(*, total_sales: float | None = None, total_profit: float | None = None, anomaly_count: int | None = None, anomaly_pct: float | None = None, rfm_summary: dict[str, Any] | None = None, forecast: ForecastArtifacts | None = None, anomalies: AnomalyArtifacts | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if total_sales is not None:
        summary["total_sales"] = total_sales
    if total_profit is not None:
        summary["total_profit"] = total_profit
    if anomaly_count is not None:
        summary["anomaly_count"] = anomaly_count
    if anomaly_pct is not None:
        summary["anomaly_pct"] = anomaly_pct
    if rfm_summary is not None:
        summary["rfm"] = rfm_summary
    if forecast is not None:
        summary["forecast_metrics"] = forecast.metrics
        summary["forecast_top_feature"] = forecast.top_feature
    if anomalies is not None:
        summary["anomaly_top_features"] = anomalies.top_features
    return summary
