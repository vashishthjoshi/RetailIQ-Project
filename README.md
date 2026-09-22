# 📊 RetailIQ — AI-Powered Retail Business Intelligence & Sales Forecasting Platform

RetailIQ is a single-file, full-stack **Streamlit** web application that turns raw, messy
retail transaction data into a cleaned dataset, an interactive Business Intelligence
dashboard, a machine-learning profit predictor, and automatically generated business
insights — all in one place.

Built as the capstone project for the **BharatCares Data Analytics Internship —
Masterclass Capstone Project**.

---

## 🔗 Dataset

**Sample Superstore Sales Dataset**
Source: https://raw.githubusercontent.com/Bimal2614/Data-Analysis/main/Super%20Store%20Dataset.csv

The app loads the dataset automatically in this order of priority:
1. A CSV uploaded by the user via the sidebar file uploader
2. A local copy at `data/Superstore.csv` (if present in the project folder)
3. The public source link above (fetched automatically if no local file is found)

This means the app runs out-of-the-box with **no manual data download required**.

---

## 📝 Project Description

Retail businesses generate large volumes of transactional data that is often
incomplete, duplicated, and hard to interpret at a glance. RetailIQ addresses this
end-to-end by taking a raw superstore sales export and producing:

1. **Cleaned, analysis-ready data** — a documented, transparent cleaning pipeline that
   removes duplicates and intelligently repairs missing values (rather than blindly
   dropping rows), preserving as much usable business data as possible.
2. **An interactive BI dashboard** — KPIs (Total Sales, Total Profit, Orders, Profit
   Margin, Average Discount) plus exploratory charts covering regional performance,
   category/sub-category sales, discount-vs-profit relationships, and fulfillment
   times.
3. **A machine-learning profit predictor** — a Random Forest Regressor trained to
   predict profit margin from order attributes (sales value, quantity, discount,
   category, region, segment, ship mode), paired with a "What-If Simulator" so a
   pricing or merchandising team can estimate the profit of a hypothetical order
   before finalizing a discount decision.
4. **Automatically generated business insights** — plain-language, data-driven
   insights (best/worst performing regions, loss-making sub-categories, the impact of
   discounting on margin, seasonal sales peaks, etc.) that recompute live as dashboard
   filters change.

The project follows the **Data → Information → Insight → Decision → Action**
business intelligence framework from raw records through to actionable
recommendations.

---

## 🛠️ Technologies Used

| Category            | Tools / Libraries                          |
|----------------------|---------------------------------------------|
| Language             | Python 3.10+                                |
| Web app framework    | Streamlit                                   |
| Data handling        | Pandas, NumPy                               |
| Visualization        | Plotly Express, Plotly Graph Objects        |
| Machine learning      | scikit-learn (RandomForestRegressor, Pipeline, ColumnTransformer, OneHotEncoder) |
| Model evaluation     | scikit-learn metrics (R², MAE, RMSE)        |
| Trendline statistics | statsmodels (used internally by Plotly's OLS trendlines) |

---

## 📂 Project Structure

```
RetailIQ-Project/
│
├── app.py                # Main Streamlit application (single-file full-stack app)
├── requirements.txt      # Python dependencies
├── README.md              # Project overview (this file)
├── data/                  # (optional) place a local Superstore.csv here
└── YourName_ProjectReport.docx   # Full project documentation
```

---

## ⚙️ Setup / Run Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/RetailIQ-Project.git
cd RetailIQ-Project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> No dataset download is required — the app fetches the Superstore dataset
> automatically from the source link above the first time it runs. You can also
> upload your own Superstore-format CSV from the sidebar.

---

## 🚀 Key Features

- 🧹 **Transparent data cleaning pipeline** with a before/after data-quality report
- 📈 **Interactive KPI dashboard** with dynamic sidebar filters (Year, Region, Category)
- 🤖 **AI profit-prediction model** with feature importance and predicted-vs-actual evaluation
- 🔮 **What-If Simulator** to test pricing/discount scenarios before placing an order
- 💡 **Auto-generated, filter-aware business insights**
- 🎨 Clean, custom-styled UI built entirely with Streamlit

---

## 👤 Author

**Vashishth**
BharatCares Data Analytics Internship — Masterclass Capstone Project

---

## 📄 License

This project was created for educational purposes as part of the BharatCares Data
Analytics Internship. Free to use for learning and reference.
