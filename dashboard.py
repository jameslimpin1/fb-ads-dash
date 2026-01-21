import streamlit as st
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, date

# --- Page Configuration ---
st.set_page_config(
    page_title="🚀 2025 FB Ads Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
REPORT_FILE = "Final_Matched_FB_Voluum_Report.csv"
NUMERIC_COLUMNS = ['Conversions', 'Normalized_Spend_USD', 'Visits', 'Unique visits', 'Unique Visit %', 'CPA', 'CV']
CHART_HEIGHT = 600

@st.cache_data
def load_data():
    if not os.path.exists(REPORT_FILE):
        return None
    
    # 1. Load Data (Force Ad ID to string to keep "1202..." intact)
    df = pd.read_csv(REPORT_FILE, dtype={'Ad ID': str})
    
    # 2. Filter Garbage (Remove template tags or empty IDs)
    df = df.dropna(subset=['Ad ID'])
    df = df[~df['Ad ID'].astype(str).str.contains('{', na=False)]
    df = df[df['Ad ID'] != 'nan']

    # 3. Date Sanitization
    # We prefer 'Day' as this is a Daily Report
    if 'Day' in df.columns:
        df['Date'] = pd.to_datetime(df['Day'], errors='coerce')
    else:
        df['Date'] = pd.to_datetime(df['Month Date'], errors='coerce')

    df['Month Date'] = pd.to_datetime(df['Month Date'], errors='coerce')
    df = df.dropna(subset=['Date']) 
    
    # 4. Numeric Conversion & Cleanup
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # FIX: Convert float-like integers (60.0 -> 60) for cleaner display
    if 'Conversions' in df.columns:
        df['Conversions'] = df['Conversions'].astype(int)
    if 'Visits' in df.columns:
        df['Visits'] = df['Visits'].astype(int)
    if 'Unique visits' in df.columns:
        df['Unique visits'] = df['Unique visits'].astype(int)

    # 5. Percentage Scaling (Handle 0.90 vs 90.0)
    if df['Unique Visit %'].max() <= 1.1:
        df['Unique Visit %'] = df['Unique Visit %'] * 100
    
    # 6. String Safety (Prevent TypeError)
    df['Facebook Video Link'] = df['Facebook Video Link'].fillna("").astype(str)
    df['Ad name'] = df['Ad name'].fillna(df['Ad ID']).astype(str)
    
    # 7. Clickable Labels for Charts
    df['Clickable_Label'] = np.where(
        df['Facebook Video Link'].str.contains('http', na=False),
        '<a href="' + df['Facebook Video Link'] + '" target="_blank" style="color: #00CC96; text-decoration: none;">' + df['Ad name'] + '</a>',
        df['Ad name']
    )
    return df

def display_kpi_metrics(df):
    # --- KPI AGGREGATION ---
    t_spend = df['Normalized_Spend_USD'].sum()
    t_convs = df['Conversions'].sum()
    t_visits = df['Visits'].sum()
    t_u_visits = df['Unique visits'].sum()
    
    # Weighted CPA Calculation
    total_payout = (df['CPA'] * df['Conversions']).sum()
    avg_cpa = total_payout / t_convs if t_convs > 0 else 0

    u_pct = (t_u_visits / t_visits * 100) if t_visits > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Spend", f"${t_spend:,.0f}", help="Sum of daily spend")
    c2.metric("Conversions", f"{t_convs:,.0f}")
    c3.metric("Weighted Avg CPA", f"${avg_cpa:.2f}")
    c4.metric("Unique Visits", f"{t_u_visits:,.0f}")
    c5.metric("Unique %", f"{u_pct:.1f}%")

def main():
    raw_df = load_data()
    if raw_df is None:
        st.error("❌ Data file not found. Please run the report generator.")
        return
    
    # --- 1. SIDEBAR ---
    st.sidebar.header("🕹️ Control Panel")
    view_option = st.sidebar.radio(
        "🔎 View",
        options=["Standard Dashboard", "Focus: Trends", "Focus: Funnel", "Focus: Performance Chart"],
        index=0
    )
    st.sidebar.divider()

    # --- 2. DATE FILTERS ---
    st.sidebar.write("📅 **Select Date Range**")
    min_d, max_d = raw_df['Date'].min().date(), raw_df['Date'].max().date()
    default_s, default_e = date(2025, 1, 1), date(2025, 12, 31)

    with st.sidebar.form("date_filter"):
        date_range = st.date_input("Range", (default_s, default_e), min_value=min_d, max_value=max_d)
        submit = st.form_submit_button("Apply")

    if 'dates' not in st.session_state:
        st.session_state.dates = (default_s, default_e)
    if submit and len(date_range) == 2:
        st.session_state.dates = date_range

    start_date, end_date = st.session_state.dates
    df = raw_df[(raw_df['Date'].dt.date >= start_date) & (raw_df['Date'].dt.date <= end_date)]

    # Search Filter
    search_q = st.sidebar.text_input("🔍 Search Ad Name/ID:")
    if search_q:
        df = df[df['Ad name'].str.contains(search_q, case=False) | df['Ad ID'].str.contains(search_q, case=False)]

    # --- DYNAMIC ROW COUNTS ---
    # Calculate how many unique ads matched the filters
    total_ads_matched = df['Ad ID'].nunique()
    
    # Ensure limits are safe (at least 1)
    max_limit = max(1, total_ads_matched)
    # Default value shouldn't exceed the max available
    default_val = min(20, max_limit)

    row_count_td = st.sidebar.number_input(
        "🔢 Monthly/Daily Breakdown Rows:", 
        min_value=1, 
        max_value=max_limit, 
        value=default_val,
        help=f"Max available: {max_limit}"
    )
    
    row_count_lb = st.sidebar.number_input(
        "🔢 Master Leaderboard Rows:", 
        min_value=1, 
        max_value=max_limit, 
        value=default_val,
        help=f"Max available: {max_limit}"
    )

    # --- 3. MAIN DASHBOARD ---
    st.title("🚀 2025 FB Ads Performance")
    display_kpi_metrics(df)
    st.divider()

    # --- TRENDS ---
    if view_option in ["Standard Dashboard", "Focus: Trends"]:
        st.subheader("📈 Conversion Trend")
        top_ids = df.groupby('Ad ID')['Conversions'].sum().nlargest(15).index
        trend_df = df[df['Ad ID'].isin(top_ids)].groupby(['Date', 'Ad name'])['Conversions'].sum().reset_index()
        
        h = 700 if "Focus" in view_option else 400
        fig = px.line(trend_df, x="Date", y="Conversions", color="Ad name", markers=True, template="plotly_dark", height=h)
        st.plotly_chart(fig, width="stretch")

    # --- BREAKDOWN TABLES ---
    if view_option in ["Standard Dashboard", "Focus: Trends"]:
        tab_mo, tab_da = st.tabs(["📅 Monthly Breakdown", "🗓️ Daily Breakdown"])
        
        # Calculate total unique ads in the current filter context
        total_ads_in_view = df['Ad ID'].nunique()
        
        # Monthly Pivot
        with tab_mo:
            if not df.empty:
                p_mo = df.pivot_table(index=['Ad ID', 'Ad name', 'Facebook Video Link'], columns='Month Date', values='Conversions', aggfunc='sum', fill_value=0)
                p_mo.columns = [c.strftime('%b') for c in p_mo.columns]
                p_mo['Total'] = p_mo.sum(axis=1)
                
                # Sort and Limit by row_count_td
                p_mo = p_mo.sort_values('Total', ascending=False).head(int(row_count_td)).reset_index()
                
                # Reorder columns
                cols = ['Ad ID', 'Ad name', 'Facebook Video Link', 'Total'] + [c for c in p_mo.columns if c not in ['Ad ID', 'Ad name', 'Facebook Video Link', 'Total']]
                
                st.dataframe(
                    p_mo[cols].style.background_gradient(cmap='YlGn', subset=cols[4:], axis=1),
                    width="stretch",
                    column_config={
                        "Facebook Video Link": st.column_config.LinkColumn("Link", display_text="View"),
                        "Total": st.column_config.NumberColumn("Total", format="%d")
                    },
                    hide_index=True
                )
                # Markdown Annotation for Monthly
                st.markdown(f"**Showing top {len(p_mo)} of {total_ads_in_view} ads.**")

        # Daily Pivot
        with tab_da:
            if not df.empty:
                p_da = df.pivot_table(index=['Ad ID', 'Ad name', 'Facebook Video Link'], columns='Date', values='Conversions', aggfunc='sum', fill_value=0)
                p_da.columns = [c.strftime('%m/%d') for c in p_da.columns]
                p_da['Total'] = p_da.sum(axis=1)
                
                # Sort and Limit by row_count_td
                p_da = p_da.sort_values('Total', ascending=False).head(int(row_count_td))
                
                # Freeze Total Column
                p_da.set_index('Total', append=True, inplace=True)
                
                st.dataframe(
                    p_da,
                    width="stretch",
                    column_config={"Facebook Video Link": st.column_config.LinkColumn("Link", display_text="View")}
                )
                # Markdown Annotation for Daily
                st.markdown(f"**Showing top {len(p_da)} of {total_ads_in_view} ads.**")

    if view_option == "Standard Dashboard":
        st.divider()

    # --- AGGREGATION FOR CHARTS ---
    # Calculate Total Payout for Weighted CPA
    df['Total_Payout'] = df['CPA'] * df['Conversions']
    
    agg_df = df.groupby(['Ad ID', 'Ad name', 'Clickable_Label', 'Facebook Video Link']).agg({
        'Conversions': 'sum', 
        'Visits': 'sum', 
        'Unique visits': 'sum', 
        'Normalized_Spend_USD': 'sum', 
        'Total_Payout': 'sum'
    }).reset_index()
    
    # Weighted CPA
    agg_df['CPA'] = agg_df['Total_Payout'] / agg_df['Conversions']
    agg_df['CPA'] = agg_df['CPA'].fillna(0).replace([np.inf, -np.inf], 0)

    # --- FUNNEL ---
    if view_option in ["Standard Dashboard", "Focus: Funnel"]:
        col_a, col_b = st.columns(2) if view_option == "Standard Dashboard" else (st.container(), st.container())
        with col_a:
            st.subheader("🔥 Integrated Conversion Funnel")
            top_f = agg_df.nlargest(15, 'Conversions').sort_values('Conversions')
            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(y=top_f['Clickable_Label'], x=top_f['Visits'], name='Visits', orientation='h', marker_color='#636EFA'))
            fig_f.add_trace(go.Bar(y=top_f['Clickable_Label'], x=top_f['Unique visits'], name='Unique', orientation='h', marker_color='#00CC96'))
            fig_f.add_trace(go.Bar(y=top_f['Clickable_Label'], x=top_f['Conversions'], name='Convs', orientation='h', marker_color='#EF553B'))
            h = 800 if "Focus" in view_option else CHART_HEIGHT
            fig_f.update_layout(barmode='overlay', template="plotly_dark", height=h)
            st.plotly_chart(fig_f, width="stretch")

    # --- SCATTER (Conversions vs CPA) ---
    if view_option in ["Standard Dashboard", "Focus: Performance Chart"]:
        container = col_b if view_option == "Standard Dashboard" else st.container()
        with container:
            st.subheader("🏆 Efficiency: Conversions vs CPA")
            top_p = agg_df.nlargest(15, 'Conversions').sort_values('Conversions')
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(y=top_p['Clickable_Label'], x=top_p['Conversions'], name='Convs', orientation='h', marker_color='#00CC96'))
            fig_p.add_trace(go.Scatter(y=top_p['Clickable_Label'], x=top_p['CPA'], name='CPA ($)', mode='markers+lines', marker_color='#FFA15A', xaxis='x2'))
            h = 800 if "Focus" in view_option else CHART_HEIGHT
            fig_p.update_layout(template="plotly_dark", height=h, xaxis=dict(title="Conversions"), xaxis2=dict(title="CPA ($)", overlaying='x', side='top'))
            st.plotly_chart(fig_p, width="stretch")

    if view_option == "Standard Dashboard":
        st.divider()

    # --- LEADERBOARD ---
    st.header("📋 Master Leaderboard")
    # Apply row_count_lb
    lb = agg_df.sort_values('Conversions', ascending=False).head(int(row_count_lb))
    lb.insert(0, 'Rank', range(1, len(lb) + 1))
    
    max_c = int(agg_df['Conversions'].max()) if not agg_df.empty else 100
    max_s = int(agg_df['Normalized_Spend_USD'].max()) if not agg_df.empty else 100

    st.dataframe(
        lb[['Rank', 'Ad ID', 'Ad name', 'Conversions', 'Normalized_Spend_USD', 'CPA', 'Facebook Video Link']],
        column_config={
            "Conversions": st.column_config.ProgressColumn("Convs", format="%d", min_value=0, max_value=max_c, color="green"),
            "Normalized_Spend_USD": st.column_config.ProgressColumn("Spend ($)", format="$%.2f", min_value=0, max_value=max_s, color="yellow"),
            "CPA": st.column_config.NumberColumn("CPA ($)", format="$%.2f"),
            "Facebook Video Link": st.column_config.LinkColumn("Ad", display_text="📺 View")
        },
        hide_index=True,
        width="stretch"
    )
    # Annotation
    st.markdown(f"**Showing top {len(lb)} of {len(agg_df)} ads.**")

if __name__ == "__main__":
    main()