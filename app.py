"""
RetailIQ — AI-Powered Retail Business Intelligence & Sales Forecasting Platform
================================================================================

A single-file, full-stack (backend + frontend) Streamlit application that
converts raw, messy superstore transaction data into:

    1. Cleaned, analysis-ready data (documented cleaning pipeline)
    2. An interactive Business Intelligence dashboard (KPIs + EDA charts)
    3. A machine-learning model that predicts order Profit, plus a
       "what-if" simulator for pricing / discount decisions
    4. Automatically generated, data-driven business insights

Author : Vashishth
Project: BharatCares Data Analytics Internship — Masterclass Capstone Project
Dataset: Sample Superstore Sales Dataset (see README.md for source link)

Run with:
    streamlit run app.py
"""

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailIQ | Retail Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; color: #555; }
    h1, h2, h3 { color: #1f2c56; }
    .insight-card {
        background-color: #f4f7ff;
        border-left: 5px solid #3b5bdb;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .footer-note { color: #888; font-size: 0.8rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "data", "Superstore.csv"),
    "data/Superstore.csv",
    "Superstore.csv",
]
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/Bimal2614/Data-Analysis/main/"
    "Super%20Store%20Dataset.csv"
)

# --------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data(uploaded_file=None) -> pd.DataFrame:
    """Load the raw Superstore dataset from an uploaded file, a bundled
    local copy, or (as a last resort) directly from its public GitHub
    source, so the app runs out-of-the-box in almost any environment."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    for path in DATA_PATH_CANDIDATES:
        if os.path.exists(path):
            return pd.read_csv(path)

    # Fall back to fetching the dataset from its public source
    return pd.read_csv(DATASET_SOURCE_URL)


@st.cache_data(show_spinner=False)
def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full data-cleaning pipeline. Returns the cleaned dataframe plus a
    report dict describing exactly what was fixed (used in the
    'Data Quality' tab and the project report)."""
    df = raw.copy()
    report = {}

    report["rows_before"] = len(df)
    report["missing_before"] = int(df.isna().sum().sum())

    # 1. Drop fully duplicated rows (ignoring the row-id primary key)
    id_col = "Row ID" if "Row ID" in df.columns else None
    dedupe_cols = [c for c in df.columns if c != id_col]
    dup_mask = df.duplicated(subset=dedupe_cols)
    report["duplicates_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    # 2. Parse dates
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3. Impute missing Category using the (deterministic) Sub-Category ->
    #    Category mapping learned from the rest of the data instead of
    #    dropping the rows.
    if {"Category", "Sub-Category"}.issubset(df.columns):
        cat_map = (
            df.dropna(subset=["Category"])
            .groupby("Sub-Category")["Category"]
            .agg(lambda s: s.mode().iat[0])
        )
        missing_cat = df["Category"].isna()
        report["category_imputed"] = int(missing_cat.sum())
        df.loc[missing_cat, "Category"] = df.loc[missing_cat, "Sub-Category"].map(cat_map)

    # 4. Impute missing Quantity / Sales / Profit using the median value
    #    (or median profit MARGIN, for profit) within the same Sub-Category,
    #    which preserves realistic, category-specific business patterns
    #    far better than a single global average would.
    if "Quantity" in df.columns:
        missing_qty = df["Quantity"].isna()
        report["quantity_imputed"] = int(missing_qty.sum())
        med_qty = df.groupby("Sub-Category")["Quantity"].transform("median")
        df["Quantity"] = df["Quantity"].fillna(med_qty)

    if "Sales" in df.columns:
        missing_sales = df["Sales"].isna()
        report["sales_imputed"] = int(missing_sales.sum())
        med_sales = df.groupby("Sub-Category")["Sales"].transform("median")
        df["Sales"] = df["Sales"].fillna(med_sales)

    if "Profit" in df.columns and "Sales" in df.columns:
        missing_profit = df["Profit"].isna()
        report["profit_imputed"] = int(missing_profit.sum())
        margin = (df["Profit"] / df["Sales"]).replace([np.inf, -np.inf], np.nan)
        med_margin = margin.groupby(df["Sub-Category"]).transform("median")
        estimated_profit = df["Sales"] * med_margin
        df["Profit"] = df["Profit"].fillna(estimated_profit)

    # 5. Drop any still-unresolvable rows (rare edge cases with no
    #    Sub-Category peers to borrow statistics from)
    core_cols = [c for c in ["Category", "Sales", "Quantity", "Profit"] if c in df.columns]
    before_drop = len(df)
    df = df.dropna(subset=core_cols)
    report["unresolvable_rows_dropped"] = before_drop - len(df)

    # 6. Correct data types
    if "Quantity" in df.columns:
        df["Quantity"] = df["Quantity"].round().astype(int)
    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].astype(str)

    # 7. Feature engineering used throughout the dashboard & model
    if "Order Date" in df.columns:
        df["Order Year"] = df["Order Date"].dt.year
        df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
        df["Order Month Name"] = df["Order Date"].dt.strftime("%b")
    if {"Sales", "Profit"}.issubset(df.columns):
        raw_margin = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], 0)
        # Clip the handful of extreme outlier ratios (near-zero Sales with a
        # large Profit/Loss) so they don't dominate the ML model below;
        # affects <0.5% of rows.
        df["Profit Margin"] = np.clip(raw_margin, -3, 1)
    if "Order Date" in df.columns and "Ship Date" in df.columns:
        df["Fulfillment Days"] = (df["Ship Date"] - df["Order Date"]).dt.days

    report["rows_after"] = len(df)
    report["missing_after"] = int(df.isna().sum().sum())
    return df, report


# --------------------------------------------------------------------------
# MACHINE LEARNING — PROFIT PREDICTION MODEL
# --------------------------------------------------------------------------
FEATURE_NUMERIC = ["Sales", "Quantity", "Discount"]
FEATURE_CATEGORICAL = ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"]
# The model predicts PROFIT MARGIN (Profit / Sales) rather than raw Profit.
# Raw order profit is extremely heavy-tailed (a handful of bulk Machines /
# Copiers orders swing thousands of dollars either way), which makes it an
# unstable regression target. Margin is bounded, far more consistent across
# orders of very different sizes, and it is also the more *actionable*
# number for a pricing/discount decision. Profit is then derived as
# predicted_margin * Sales for reporting and the what-if simulator.
TARGET = "Profit Margin"


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    features = FEATURE_NUMERIC + FEATURE_CATEGORICAL
    data = df.dropna(subset=features + [TARGET, "Profit"])
    X = data[features]
    y = data[TARGET]
    sales_full = data["Sales"]
    profit_full = data["Profit"]

    (
        X_train, X_test,
        y_train, y_test,
        sales_train, sales_test,
        profit_train, profit_test,
    ) = train_test_split(
        X, y, sales_full, profit_full, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURE_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CATEGORICAL),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    margin_pred = model.predict(X_test)
    profit_pred = margin_pred * sales_test.values

    metrics = {
        "margin_r2": r2_score(y_test, margin_pred),
        "margin_mae": mean_absolute_error(y_test, margin_pred),
        "profit_r2": r2_score(profit_test, profit_pred),
        "mae": mean_absolute_error(profit_test, profit_pred),
        "rmse": root_mean_squared_error(profit_test, profit_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Feature importance (mapped back to readable names)
    ohe = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(FEATURE_CATEGORICAL))
    all_names = FEATURE_NUMERIC + cat_names
    importances = model.named_steps["regressor"].feature_importances_
    fi = (
        pd.DataFrame({"feature": all_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(12)
    )

    test_results = X_test.copy()
    test_results["Actual Profit"] = profit_test.values
    test_results["Predicted Profit"] = profit_pred
    test_results["Actual Margin"] = y_test.values
    test_results["Predicted Margin"] = margin_pred

    return model, metrics, fi, test_results


# --------------------------------------------------------------------------
# AUTOMATED BUSINESS INSIGHT GENERATION
# --------------------------------------------------------------------------
def generate_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    region_profit = df.groupby("Region")["Profit"].sum().sort_values(ascending=False)
    insights.append(
        f"**{region_profit.index[0]}** is the most profitable region "
        f"(₹/$ {region_profit.iloc[0]:,.0f} total profit), while "
        f"**{region_profit.index[-1]}** trails with "
        f"{region_profit.iloc[-1]:,.0f} — a strong candidate for a "
        f"regional pricing or logistics review."
    )

    cat_profit = df.groupby("Category")["Profit"].sum().sort_values()
    worst_cat = cat_profit.index[0]
    insights.append(
        f"**{cat_profit.index[-1]}** drives the most profit overall, but "
        f"**{worst_cat}** has the weakest total profit contribution "
        f"({cat_profit.iloc[0]:,.0f}) despite continued sales volume — "
        f"worth auditing for over-discounting."
    )

    disc_bins = pd.cut(
        df["Discount"], bins=[-0.01, 0, 0.1, 0.2, 0.3, 1],
        labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"],
    )
    margin_by_disc = df.groupby(disc_bins, observed=True)["Profit Margin"].mean()
    high_disc_margin = margin_by_disc.get(">30%", np.nan)
    if pd.notna(high_disc_margin):
        insights.append(
            f"Average profit margin collapses to **{high_disc_margin:.0%}** "
            f"on orders discounted more than 30%, compared with "
            f"**{margin_by_disc.get('0%', 0):.0%}** on full-price orders — "
            f"discounting policy is a direct lever on profitability."
        )

    sub_profit = df.groupby("Sub-Category")["Profit"].sum().sort_values()
    loss_subs = sub_profit[sub_profit < 0]
    if len(loss_subs) > 0:
        insights.append(
            f"**{', '.join(loss_subs.index[:3])}** are the only sub-categories "
            f"operating at a net loss overall ({loss_subs.sum():,.0f} combined) "
            f"— these need pricing correction or should be de-prioritized in "
            f"inventory planning."
        )

    if "Order Month Name" in df.columns:
        month_sales = (
            df.groupby("Order Month Name")["Sales"].sum().reindex(
                ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            ).dropna()
        )
        peak_month = month_sales.idxmax()
        insights.append(
            f"Sales consistently peak in **{peak_month}** — a clear seasonal "
            f"pattern that should drive inventory build-up and staffing "
            f"decisions ahead of that period."
        )

    seg_profit = df.groupby("Segment")["Profit"].mean().sort_values(ascending=False)
    insights.append(
        f"The **{seg_profit.index[0]}** segment yields the highest average "
        f"profit per order (${seg_profit.iloc[0]:,.2f}), suggesting "
        f"retention and upsell efforts should prioritize this segment."
    )

    return insights


# --------------------------------------------------------------------------
# LOAD + CLEAN DATA (shared across the whole app)
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 RetailIQ")
    st.caption("AI-Powered Retail Business Intelligence Platform")
    st.markdown("---")
    uploaded = st.file_uploader(
        "Optional: use your own Superstore-format CSV", type=["csv"]
    )
    st.markdown("---")

raw_df = load_raw_data(uploaded)
clean_df, clean_report = clean_data(raw_df)

with st.sidebar:
    st.subheader("Filters")
    years = sorted(clean_df["Order Year"].dropna().unique().tolist())
    selected_years = st.multiselect("Order Year", years, default=years)
    regions = sorted(clean_df["Region"].unique().tolist())
    selected_regions = st.multiselect("Region", regions, default=regions)
    categories = sorted(clean_df["Category"].unique().tolist())
    selected_categories = st.multiselect("Category", categories, default=categories)

    st.markdown("---")
    st.markdown(
        '<p class="footer-note">Dataset: Sample Superstore Sales Dataset.<br>'
        "See README.md for the full source link.</p>",
        unsafe_allow_html=True,
    )

mask = (
    clean_df["Order Year"].isin(selected_years)
    & clean_df["Region"].isin(selected_regions)
    & clean_df["Category"].isin(selected_categories)
)
view = clean_df[mask].copy()

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("📊 RetailIQ — Retail Business Intelligence & Sales Forecasting")
st.markdown(
    "Turning raw superstore transaction data into KPIs, insights, and an "
    "AI-driven profit forecast that supports real pricing and inventory "
    "decisions."
)

tab_overview, tab_quality, tab_eda, tab_model, tab_insights, tab_about = st.tabs(
    ["🏠 Overview", "🧹 Data Quality", "📈 Exploratory Analysis",
     "🤖 Profit Predictor", "💡 Business Insights", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# TAB 1 — OVERVIEW / KPIs
# --------------------------------------------------------------------------
with tab_overview:
    if view.empty:
        st.warning("No data matches the selected filters.")
    else:
        total_sales = view["Sales"].sum()
        total_profit = view["Profit"].sum()
        total_orders = view["Order ID"].nunique()
        margin = total_profit / total_sales if total_sales else 0
        avg_discount = view["Discount"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Sales", f"${total_sales:,.0f}")
        c2.metric("Total Profit", f"${total_profit:,.0f}")
        c3.metric("Orders", f"{total_orders:,}")
        c4.metric("Profit Margin", f"{margin:.1%}")
        c5.metric("Avg. Discount", f"{avg_discount:.1%}")

        st.markdown("### Monthly Sales vs Profit Trend")
        monthly = (
            view.groupby("Order Month")[["Sales", "Profit"]].sum().reset_index()
            .sort_values("Order Month")
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["Order Month"], y=monthly["Sales"],
                                  name="Sales", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=monthly["Order Month"], y=monthly["Profit"],
                                  name="Profit", mode="lines+markers"))
        fig.update_layout(height=420, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### Sales by Region")
            reg = view.groupby("Region", as_index=False)["Sales"].sum()
            st.plotly_chart(
                px.pie(reg, names="Region", values="Sales", hole=0.45),
                use_container_width=True,
            )
        with colB:
            st.markdown("### Sales by Category")
            cat = view.groupby("Category", as_index=False)["Sales"].sum()
            st.plotly_chart(
                px.bar(cat, x="Category", y="Sales", color="Category", text_auto=".2s"),
                use_container_width=True,
            )

# --------------------------------------------------------------------------
# TAB 2 — DATA QUALITY / CLEANING REPORT
# --------------------------------------------------------------------------
with tab_quality:
    st.markdown("### Raw Data → Clean Data")
    st.write(
        "The raw export contains realistic data-quality issues: duplicate "
        "records and scattered missing values across `Category`, `Sales`, "
        "`Quantity`, and `Profit`. Rather than dropping every incomplete "
        "row (which would discard usable information), the pipeline below "
        "repairs what can be reliably inferred and only removes rows that "
        "genuinely cannot be fixed."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before cleaning", f"{clean_report['rows_before']:,}")
    c2.metric("Duplicate rows removed", f"{clean_report['duplicates_removed']:,}")
    c3.metric("Rows after cleaning", f"{clean_report['rows_after']:,}")

    st.markdown("#### Missing values repaired")
    repair_table = pd.DataFrame(
        {
            "Field": ["Category", "Quantity", "Sales", "Profit"],
            "Missing values found": [
                clean_report.get("category_imputed", 0),
                clean_report.get("quantity_imputed", 0),
                clean_report.get("sales_imputed", 0),
                clean_report.get("profit_imputed", 0),
            ],
            "Imputation method": [
                "Mapped deterministically from Sub-Category",
                "Median Quantity within the same Sub-Category",
                "Median Sales within the same Sub-Category",
                "Sales × median profit-margin of the Sub-Category",
            ],
        }
    )
    st.dataframe(repair_table, use_container_width=True, hide_index=True)
    st.caption(
        f"{clean_report.get('unresolvable_rows_dropped', 0)} rows had no "
        "usable peer group and were dropped as a last resort. Total missing "
        f"cells: {clean_report['missing_before']} → {clean_report['missing_after']}."
    )

    st.markdown("#### Sample of cleaned data")
    st.dataframe(clean_df.head(20), use_container_width=True)

# --------------------------------------------------------------------------
# TAB 3 — EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------
with tab_eda:
    if view.empty:
        st.warning("No data matches the selected filters.")
    else:
        st.markdown("### Discount vs. Profit — where discounting hurts")
        st.plotly_chart(
            px.scatter(
                view.sample(min(2000, len(view)), random_state=1),
                x="Discount", y="Profit", color="Category",
                trendline="ols", opacity=0.6,
            ),
            use_container_width=True,
        )

        colA, colB = st.columns(2)
        with colA:
            st.markdown("### Top 10 Sub-Categories by Sales")
            top_sub = (
                view.groupby("Sub-Category", as_index=False)["Sales"]
                .sum().sort_values("Sales", ascending=False).head(10)
            )
            st.plotly_chart(
                px.bar(top_sub, x="Sales", y="Sub-Category", orientation="h"),
                use_container_width=True,
            )
        with colB:
            st.markdown("### Profit by Customer Segment")
            seg = view.groupby("Segment", as_index=False)["Profit"].sum()
            st.plotly_chart(
                px.bar(seg, x="Segment", y="Profit", color="Segment"),
                use_container_width=True,
            )

        st.markdown("### Top 10 Most Profitable vs. Loss-Making Products")
        prod_profit = view.groupby("Product Name")["Profit"].sum()
        top_products = prod_profit.sort_values(ascending=False).head(10)
        bottom_products = prod_profit.sort_values().head(10)
        colC, colD = st.columns(2)
        with colC:
            st.markdown("**Most profitable**")
            st.dataframe(top_products.reset_index().rename(columns={0: "Profit"}),
                         use_container_width=True, hide_index=True)
        with colD:
            st.markdown("**Biggest losses**")
            st.dataframe(bottom_products.reset_index().rename(columns={0: "Profit"}),
                         use_container_width=True, hide_index=True)

        st.markdown("### Average Fulfillment Time by Ship Mode")
        ship = view.groupby("Ship Mode", as_index=False)["Fulfillment Days"].mean()
        st.plotly_chart(
            px.bar(ship, x="Ship Mode", y="Fulfillment Days", color="Ship Mode"),
            use_container_width=True,
        )

# --------------------------------------------------------------------------
# TAB 4 — PREDICTIVE MODEL (PROFIT PREDICTOR)
# --------------------------------------------------------------------------
with tab_model:
    st.markdown("### AI Profit Prediction Model")
    st.write(
        "A Random Forest Regressor is trained on the cleaned transaction "
        "data to predict the **Profit** of an order from its sales value, "
        "quantity, discount, and product/customer attributes. This powers "
        "the what-if simulator below, which a merchandising or pricing "
        "team can use before finalising a discount decision."
    )

    with st.spinner("Training model..."):
        model, metrics, feat_importance, test_results = train_model(clean_df)

    st.caption(
        "The model is trained to predict **profit margin** (Profit ÷ Sales) "
        "rather than raw profit dollars, because a handful of very large "
        "bulk orders make raw profit an unstable, heavy-tailed target. "
        "Predicted profit is then derived as *predicted margin × Sales*, "
        "which is also the more actionable number for a pricing decision."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Margin R² (model fit)", f"{metrics['margin_r2']:.3f}")
    c2.metric("Derived Profit R²", f"{metrics['profit_r2']:.3f}")
    c3.metric("Profit MAE", f"${metrics['mae']:.2f}")
    c4.metric("Profit RMSE", f"${metrics['rmse']:.2f}")
    st.caption(
        f"Trained on {metrics['n_train']:,} orders, evaluated on "
        f"{metrics['n_test']:,} held-out orders (80/20 split)."
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Feature Importance")
        st.plotly_chart(
            px.bar(feat_importance, x="importance", y="feature", orientation="h"),
            use_container_width=True,
        )
    with colB:
        st.markdown("#### Predicted vs. Actual Profit (test set)")
        st.plotly_chart(
            px.scatter(
                test_results, x="Actual Profit", y="Predicted Profit",
                opacity=0.5, trendline="ols",
            ),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 🔮 What-If Simulator")
    st.write("Estimate the profit of a hypothetical order before you place it.")

    with st.form("whatif_form"):
        f1, f2, f3 = st.columns(3)
        with f1:
            in_category = st.selectbox("Category", sorted(clean_df["Category"].unique()))
            sub_opts = sorted(
                clean_df.loc[clean_df["Category"] == in_category, "Sub-Category"].unique()
            )
            in_subcat = st.selectbox("Sub-Category", sub_opts)
        with f2:
            in_region = st.selectbox("Region", sorted(clean_df["Region"].unique()))
            in_segment = st.selectbox("Segment", sorted(clean_df["Segment"].unique()))
        with f3:
            in_ship = st.selectbox("Ship Mode", sorted(clean_df["Ship Mode"].unique()))
            in_sales = st.number_input("Sales value ($)", min_value=1.0, value=250.0, step=10.0)

        f4, f5 = st.columns(2)
        with f4:
            in_qty = st.number_input("Quantity", min_value=1, value=2, step=1)
        with f5:
            in_discount = st.slider("Discount", 0.0, 0.8, 0.1, 0.05)

        submitted = st.form_submit_button("Predict Profit")

    if submitted:
        input_df = pd.DataFrame([{
            "Sales": in_sales, "Quantity": in_qty, "Discount": in_discount,
            "Category": in_category, "Sub-Category": in_subcat,
            "Region": in_region, "Segment": in_segment, "Ship Mode": in_ship,
        }])
        predicted_margin = model.predict(input_df)[0]
        predicted_profit = predicted_margin * in_sales

        st.success(
            f"Predicted profit: **${predicted_profit:,.2f}** "
            f"(≈ {predicted_margin:.1%} margin)"
        )
        if predicted_profit < 0:
            st.error(
                "⚠️ This configuration is predicted to run at a **loss**. "
                "Consider reducing the discount or reviewing the price point."
            )
        elif predicted_margin < 0.10:
            st.warning(
                "This order is predicted to clear a thin margin (<10%). "
                "Proceed with caution on further discounting."
            )

# --------------------------------------------------------------------------
# TAB 5 — BUSINESS INSIGHTS
# --------------------------------------------------------------------------
with tab_insights:
    st.markdown("### 💡 Automatically Generated Business Insights")
    st.write(
        "These insights are computed directly and dynamically from the "
        "current filtered dataset — not hard-coded — so they update if the "
        "sidebar filters change."
    )
    if view.empty or len(view) < 20:
        st.info("Select a broader filter range to generate reliable insights.")
    else:
        import re
        for point in generate_insights(view):
            html_point = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", point)
            st.markdown(f'<div class="insight-card">{html_point}</div>', unsafe_allow_html=True)

    st.markdown("### Business Intelligence Framework Applied")
    st.write(
        textwrap.dedent(
            """
            **Data → Information → Insight → Decision → Action**

            - **Data**: raw superstore order records (often incomplete / duplicated)
            - **Information**: cleaned, structured transaction table
            - **Insight**: KPI trends, discount-profit relationship, regional/segment performance
            - **Decision**: where to cut discounting, which regions/segments to prioritise
            - **Action**: pricing policy changes, inventory allocation, targeted retention campaigns
            """
        )
    )

# --------------------------------------------------------------------------
# TAB 6 — ABOUT
# --------------------------------------------------------------------------
with tab_about:
    st.markdown("### About this project")
    st.write(
        """
        **RetailIQ** is a capstone project built for the BharatCares Data
        Analytics Internship masterclass series. It demonstrates the full
        Business Intelligence lifecycle — data cleaning, exploratory
        analysis, dashboarding, and AI-based prediction — on a single
        combined dataset, all in one Python/Streamlit application.
        """
    )
    st.markdown("**Tech stack:** Python, Pandas, NumPy, scikit-learn, Plotly, Streamlit")
    st.markdown("**Dataset:** Sample Superstore Sales Dataset — see `README.md` for the source link.")
    st.markdown(f"App generated / last run: {datetime.now():%Y-%m-%d}")
