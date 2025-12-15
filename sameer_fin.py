import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. CONFIGURATION (Aesthetic Foundation) ---
st.set_page_config(
    page_title="Stock Simulation & Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA CONSTANTS & LOADING (Cleaned Data) ---

# Define the original file paths based on the uploaded data
HISTORICAL_FILES = {
    "Asian Paints": "stock prices.xlsx - AsianPaints.csv",
    "SBI": "stock prices.xlsx - SBI.csv",
    "Tata Steel": "stock prices.xlsx - Tata steel.csv"
}
SIMULATED_FILE = "stock prices.xlsx"

# Function to clean and process each historical stock file
def process_historical_file(file_path, stock_name):
    try:
        df = pd.read_excel(file_path)
    except Exception:
        return None

    # Cleaning: Remove footer/metadata (rows where DATE is not a date)
    if 'DATE' in df.columns:
        df['DATE_cleaned'] = pd.to_datetime(df['DATE'], errors='coerce')
        first_invalid_date_index = df[df['DATE_cleaned'].isna()].index.min()
        if not pd.isna(first_invalid_date_index):
            df = df.iloc[:first_invalid_date_index]
        df = df.drop(columns=['DATE_cleaned'])

    # Data Type Conversion
    numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', '52W H', '52W L', 'VOLUME']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['STOCK'] = stock_name
    
    # Sort for time series analysis
    return df.dropna(subset=['DATE', 'CLOSE']).sort_values(by='DATE', ascending=True)

def process_simulated_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Clean column names
        df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'PCT') for col in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        numeric_cols_sim = ['CLOSE', 'CHANGE_PCT', 'VOLUME']
        for col in numeric_cols_sim:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_all_data():
    """Loads and preprocesses all data."""
    all_historical_data = []
    for name, path in HISTORICAL_FILES.items():
        df_temp = process_historical_file(path, name)
        if df_temp is not None:
            all_historical_data.append(df_temp)
            
    df_historical = pd.concat(all_historical_data, ignore_index=True) if all_historical_data else pd.DataFrame()
    df_simulated = process_simulated_data(SIMULATED_FILE)
    
    return df_historical, df_simulated

# Load the processed data
df_historical, df_simulated = load_all_data()

if df_historical.empty or df_simulated.empty:
    st.error("❌ Data files could not be loaded. Please ensure all CSV files are in the same directory.")
    st.stop()
    
# --- 3. SIDEBAR AND FILTERS ---

st.sidebar.title("⚙️ Stock Selector")
selected_stock = st.sidebar.selectbox(
    "Choose a Stock for Detailed View:",
    options=list(HISTORICAL_FILES.keys()),
    index=0 
)
st.sidebar.markdown("---")
st.sidebar.info("This dashboard provides an in-depth view of historical price movements and the final simulated close price.")


# Filter the data for the selected stock
df_selected_hist = df_historical[df_historical['STOCK'] == selected_stock]
df_selected_sim = df_simulated[df_simulated['STOCK'] == selected_stock].iloc[0]


# --- 4. MAIN DASHBOARD HEADER & SIMULATION KPIs (Aesthetic & Immediate Insight) ---

st.title(f"📊 {selected_stock} Analysis Dashboard")
st.markdown("### Simulated Price & Performance")

simulated_price = df_selected_sim['CLOSE']
change_pct = df_selected_sim['CHANGE_PCT'] * 100 
last_hist_close = df_selected_hist['CLOSE'].iloc[-1] # Last known historical close price

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

# KPI 1: Final Simulated Price
kpi_col1.metric(
    label="Simulated Close Price (₹)", 
    value=f"₹{simulated_price:,.2f}",
    delta=f"{change_pct:+.2f}% (Simulated Change)", 
    delta_color="normal"
)

# KPI 2: Change vs. Last Historical Close
kpi_col2.metric(
    label="Absolute Change (₹)", 
    value=f"₹{simulated_price - last_hist_close:,.2f}",
    delta_color="normal"
)

# Get 52W High/Low from the *entire* historical dataset for context
w52_high = df_selected_hist['52W H'].max()
w52_low = df_selected_hist['52W L'].min()

# KPI 3: 52-Week High
kpi_col3.metric(
    label="52-Week High (₹)", 
    value=f"₹{w52_high:,.2f}",
    delta=f"Distance from High: ₹{w52_high - simulated_price:,.2f}",
    delta_color="inverse"
)

# KPI 4: 52-Week Low
kpi_col4.metric(
    label="52-Week Low (₹)", 
    value=f"₹{w52_low:,.2f}",
    delta=f"Distance from Low: ₹{simulated_price - w52_low:,.2f}",
    delta_color="normal"
)

st.markdown("---")

# --- 5. HISTORICAL PRICE ACTION (Holistic Candlestick Chart) ---

st.markdown("### Price Action & Trading Volume")
chart_col1, chart_col2 = st.columns([3, 1])

# A. CANDLESTICK CHART (Most holistic price view)
fig_candle = go.Figure(data=[go.Candlestick(
    x=df_selected_hist['DATE'],
    open=df_selected_hist['OPEN'],
    high=df_selected_hist['HIGH'],
    low=df_selected_hist['LOW'],
    close=df_selected_hist['CLOSE'],
    increasing_line_color='#00CC96',  # Green for up
    decreasing_line_color='#EF553B',  # Red for down
    name='Historical Price'
)])

# Add a horizontal line for the simulated price (a prediction line)
fig_candle.add_hline(
    y=simulated_price, 
    line_dash="dot", 
    line_color="gold",
    annotation_text=f"Simulated Target: ₹{simulated_price:,.2f}",
    annotation_position="top right"
)

fig_candle.update_layout(
    title=f'{selected_stock} Historical OHLC with Simulated Target',
    xaxis_title='Date',
    yaxis_title='Price (₹)',
    xaxis_rangeslider_visible=False, # Hide range slider for cleaner look
    template="plotly_white",
    height=500,
    margin=dict(t=50, l=10, r=10, b=10)
)
chart_col1.plotly_chart(fig_candle, use_container_width=True)

# B. VOLUME BAR CHART
fig_volume = px.bar(
    df_selected_hist, 
    x='DATE', 
    y='VOLUME', 
    title='Trading Volume',
    labels={'VOLUME': 'Volume'},
    template="plotly_white",
    height=500
)
fig_volume.update_layout(
    showlegend=False,
    yaxis_title="",
    xaxis_title="",
    margin=dict(t=50, l=10, r=10, b=10)
)
chart_col2.plotly_chart(fig_volume, use_container_width=True)


st.markdown("---")

# --- 6. CROSS-STOCK SIMULATION COMPARISON (Contextual) ---

st.markdown("### Cross-Stock Simulated Price Comparison")

# Bar chart comparing all three simulated close prices
fig_comp = px.bar(
    df_simulated.sort_values(by='CLOSE', ascending=False), # Sort for better visualization
    x='STOCK', 
    y='CLOSE', 
    color='STOCK', 
    title='Simulated Stock Closing Prices (2025-12-31)',
    labels={'CLOSE': 'Simulated Price (₹)', 'STOCK': 'Stock'},
    template="plotly_dark", # Using dark theme for aesthetic contrast
    text_auto='$.2s' # Automatically show value on bar
)
fig_comp.update_layout(
    xaxis_title="",
    yaxis_title="",
    showlegend=False,
    margin=dict(t=50, l=10, r=10, b=10)
)

st.plotly_chart(fig_comp, use_container_width=True)


# --- 7. RAW DATA (Transparency) ---
with st.expander("🔍 View Raw Data Tables"):
    raw_col1, raw_col2 = st.columns(2)
    with raw_col1:
        st.subheader("Historical Data")
        st.dataframe(df_selected_hist.sort_values(by='DATE', ascending=False).drop(columns=['STOCK']), use_container_width=True)
    with raw_col2:
        st.subheader("Simulated Data (All Stocks)")
        st.dataframe(df_simulated.drop(columns=['DATE']), use_container_width=True)
