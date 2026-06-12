import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Retail Sales Predictor", layout="centered")

st.title("Warehouse & Retail Sales Predictor")
st.markdown("Predict monthly retail sales for a liquor item based on supplier, category, and warehouse movement.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Year", options=[2017, 2018, 2019, 2020])
    st.caption("Year the sale occurred. The dataset covers 2017–2020.")

    month = st.selectbox("Month", options=list(range(1, 13)),
                         format_func=lambda x: pd.Timestamp(2020, x, 1).strftime("%B"))
    st.caption("Month of the sale. Seasonal patterns vary across item types.")

    item_type = st.selectbox("Item Type", options=["LIQUOR", "WINE", "BEER", "NON-ALCOHOL"])
    st.caption("Product category. Liquor and Beer tend to have higher average retail volumes.")

with col2:
    supplier = st.selectbox("Supplier", options=[
        "E & J GALLO WINERY", "DIAGEO NORTH AMERICA INC", "CONSTELLATION BRANDS",
        "ANHEUSER BUSCH INC", "JIM BEAM BRANDS CO", "MILLER BREWING COMPANY",
        "CROWN IMPORTS", "SAZERAC CO", "BACARDI USA INC", "THE WINE GROUP",
        "PERNOD RICARD USA LLC", "HEINEKEN USA", "HEAVEN HILL DISTILLERIES INC",
        "BROWN-FORMAN BEVERAGES WORLDWIDE", "FIFTH GENERATION INC",
        "BOSTON BEER CORPORATION", "DELICATO FAMILY VINEYARDS",
        "DEUTSCH FAMILY WINE & SPIRITS", "TREASURY WINE ESTATES AMERICAS COMPANY",
        "PROXIMO SPIRITS INC", "REPUBLIC NATIONAL DISTRIBUTING CO",
        "SOUTHERN GLAZERS WINE AND SPIRITS", "JACKSON FAMILY ENTERPRISES INC",
        "MARK ANTHONY BRANDS INC", "YUENGLING BREWERY",
        "CAMPARI AMERICA LLC", "TROEGS BREWING COMPANY",
        "SIERRA NEVADA BREWING CO", "ATLAS BREW WORKS LLC", "LEGENDS LTD",
        "OTHER"
    ])
    st.caption("Distributor supplying the product to Montgomery County. Top 30 suppliers by retail volume are listed; all others fall under 'OTHER'.")

    warehouse_sales = st.number_input("Warehouse Sales (units)", min_value=0.0,
                                      max_value=10000.0, value=10.0, step=0.5)
    st.caption("Units moved from the warehouse that month. This is the strongest continuous predictor of retail demand — higher warehouse movement typically signals higher retail sales.")

st.divider()

if st.button("Predict Retail Sales", use_container_width=True):
    data = CustomData(
        year=year,
        month=month,
        supplier=supplier,
        item_type=item_type,
        warehouse_sales=warehouse_sales,
    )
    df = data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)

    st.success(f"**Predicted Retail Sales: {prediction[0]:,.2f} units**")
    st.caption("Prediction is based on a Random Forest model trained on Montgomery County warehouse and retail sales data (2017–2020).")