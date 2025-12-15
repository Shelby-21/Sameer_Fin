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

# --- Define the static file path and expected sheet names ---
FILE_PATH = "stock_prices.xlsx" # The file must be in the same directory as this script.

# FIX: Changed "Tata Steel" sheet name to "Tata steel" (lowercase 's') to match likely Excel sheet
EXPECTED_SHEETS = {
    "Asian Paints": "AsianPaints",
    "SBI": "SBI",
    "Tata Steel": "Tata steel" 
}
SIMULATED_SHEET_NAME = "Simulated Price"


# --- 2. CORE DATA PROCESSING FUNCTIONS ---

# Function to clean and process a historical stock sheet
def process_historical_sheet(df, stock_name):
    if df.empty: return pd.DataFrame()

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
    
    return df.dropna(subset=['DATE', 'CLOSE']).sort_values(by='DATE', ascending=True)

# Function to process the simulated data sheet
def process_simulated_sheet(df):
    if df.empty: return pd.DataFrame()
    
    # Clean column names (e.g., 'CHANGE (%)' to 'CHANGE_PCT')
    df.columns = [col.strip().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'PCT') for col in df.columns]
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    
    numeric_cols_sim = ['CLOSE', 'CHANGE_PCT', 'VOLUME']
    for col in numeric_cols_sim:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['DATE', 'CLOSE'])

@st.cache_data
def load_all_data(file_path):
    """Reads all sheets from the static Excel file path."""
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame(), "File not found. Please ensure 'stock_price.xlsx' is in the same directory as the script."
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Error reading Excel file: {e}"

    all_historical_data = []
    
    # Process Historical Sheets
    for stock_key, sheet_name in EXPECTED_SHEETS.items():
        if sheet_name in all_sheets:
            df_temp = process_historical_sheet(all_sheets[sheet_name], stock_key)
            if not df_temp.empty:
                all_historical_data.append(df_temp)
        
    df_historical = pd.concat(all_historical_data, ignore_index=True) if all_historical_data else pd.DataFrame()

    # Process Simulated Sheet
    df_simulated = pd.DataFrame()
    if SIMULATED_SHEET_NAME in all_sheets:
        df_simulated = process_simulated_sheet(all_sheets[SIMULATED_SHEET_NAME])

    # Final check for critical data
    if df_historical.empty or df_simulated.empty:
        missing_sheets = [name for name, sheet in EXPECTED_SHEETS.items() if sheet not in all_sheets]
        if SIMULATED_SHEET_NAME not in all_sheets: missing_sheets.append(SIMULATED_SHEET_NAME)
        
        return pd.DataFrame(), pd.DataFrame(), f"One or more required sheets are missing or empty: {', '.join(missing_sheets)}"

    return df_historical, df_simulated, None


# --- 3. LOAD DATA AND HANDLE ERRORS ---

df_historical, df_simulated, error_message = load_all_data(FILE_PATH)

st.sidebar.title("⚙️ Stock Selector")

# Global error handling for file loading or processing
if error_message:
    st.title("Data Loading Error")
    st.error(error_message)
    st.markdown("Please verify that:")
    st.markdown(f"1. The file `{FILE_PATH}` is present in the repository.")
    st.markdown(f"2. The Excel sheets are correctly named: `{', '.join(EXPECTED_SHEETS.values())}` and `{SIMULATED_SHEET_NAME}`.")
    st.stop()
    
# --- Proceed with dashboard rendering if data is successful ---

# Stock Selector
selected_stock = st.sidebar.selectbox(
    "Choose a Stock for Detailed View:",
    options=list(EXPECTED_SHEETS.keys()),
    index=0 
)
st.sidebar.markdown("---")
st.sidebar.info("Data loaded successfully from the 'stock_price.xlsx' file in the repository.")

    
# Filter the data for the selected stock
df_selected_hist = df_historical[df_historical['STOCK'] == selected_stock]

# ERROR PREVENTION: Check if the historical data frame is empty
if df_selected_hist.empty:
    st.title(f"Data Not Found for {selected_stock}")
    st.error(f"Historical data for {selected_stock} could not be loaded or is empty after cleaning.")
    st.stop()

# This is the line that was crashing, now it's protected by the check above
last_hist_close = df_selected_hist['CLOSE'].iloc[-1] 
    
# Use .str.contains for flexible matching in simulated data (e.g., 'Tata Steel' vs 'Tata steel')
# This assumes the 'STOCK' column in the simulated data has the name string.
df_selected_sim = df_simulated[df_simulated['STOCK'].str.contains(selected_stock.split()[0], case=False, na=False)].iloc[0]


# --- 4. MAIN DASHBOARD HEADER & SIMULATION KPIs (Aesthetic & Immediate Insight) ---

st.title(f"📊 {selected_stock} Analysis Dashboard")
st.markdown("### Simulated Price & Performance")

simulated_price = df_selected_sim['CLOSE']
change_pct = df_selected_sim['CHANGE_PCT'] * 100 

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
# These lines rely on df_selected_hist not being empty, which is checked above.
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
    increasing_line_color='#00CC96',
    decreasing_line_color='#EF553B',
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
    xaxis_rangeslider_visible=False,
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

df_sim_comp = df_simulated.sort_values(by='CLOSE', ascending=False)
fig_comp = px.bar(
    df_sim_comp, 
    x='STOCK', 
    y='CLOSE', 
    color='STOCK', 
    title='Simulated Stock Closing Prices (2025-12-31)',
    labels={'CLOSE': 'Simulated Price (₹)', 'STOCK': 'Stock'},
    template="plotly_dark",
    text_auto='$.2s' 
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
