from __future__ import annotations

from typing import Any

import streamlit as st


def render_landing_page(*args: Any, **kwargs: Any) -> None:
    st.title("Retail Intelligence Dashboard with Explainable AI")
    st.subheader("Built for Superstore-style retail transactional data")
    st.write(
        "Upload an order-level CSV or Excel file with dates, sales, customers, discounts, regions, and product data."
    )


def render_dashboard(*args: Any, **kwargs: Any) -> None:
    st.write("Dashboard rendering is handled in the main app module.")


def render_chat_panel(*args: Any, **kwargs: Any) -> None:
    st.write("Chat rendering is handled in the main app module.")
