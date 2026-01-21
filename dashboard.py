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
    
    # 1. View Option
    view_option = st.sidebar.radio(
        "🔎 View",
        options=["Standard Dashboard", "Focus: Trends", "Focus: Funnel", "Focus: Performance Chart"],
        index=0,
        help="Note: The Master Leaderboard is always visible at the top. Scroll down to see your selected 'Focus' view."
    )

    st.sidebar.divider()

    # 2. Date Selection (Separate Pickers)
    st.sidebar.write("📅 **Select Date Range**")
    min_d, max_d = raw_df['Date'].min().date(), raw_df['Date'].max().date()
    default_s, default_e = date(2025, 1, 1), date(2025, 12, 31)

    # Use columns for side-by-side pickers
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", default_s, min_value=min_d, max_value=max_d)
    end_date = col2.date_input("End Date", default_e, min_value=min_d, max_value=max_d)

    if start_date > end_date:
        st.sidebar.error("Error: Start Date must be before End Date.")
        # Fallback to defaults to prevent crash
        start_date, end_date = default_s, default_e
    
    # Filter Data by Date
    df = raw_df[(raw_df['Date'].dt.date >= start_date) & (raw_df['Date'].dt.date <= end_date)]

    # 3. Master Leaderboard Rows (Before Search, max value based on Date filter only)
    total_ads_pre_search = df['Ad ID'].nunique()
    max_lb = max(1, total_ads_pre_search)
    default_lb = min(10, max_lb)
    
    row_count_lb = st.sidebar.number_input(
        "🔢 Master Leaderboard Rows:", 
        min_value=1,
        step=5, 
        max_value=max_lb, 
        value=default_lb,
        help=f"Max available (pre-search): {max_lb}"
    )

    # 4. Search Ad Name/ID
    search_q = st.sidebar.text_input("🔍 Search Ad Name/ID:")
    if search_q:
        df = df[df['Ad name'].str.contains(search_q, case=False) | df['Ad ID'].str.contains(search_q, case=False)]

    # 5. Monthly/Daily Breakdown Rows (After Search, max value based on filtered result)
    total_ads_post_search = df['Ad ID'].nunique()
    max_td = max(1, total_ads_post_search)
    default_td = min(20, max_td)

    row_count_td = st.sidebar.number_input(
        "🔢 Monthly/Daily Breakdown Rows:", 
        min_value=1,
        step=5, 
        max_value=max_td, 
        value=default_td,
        help=f"Max available: {max_td}"
    )

    # --- 3. MAIN DASHBOARD ---
    st.title("🚀 2025 FB Ads Performance")
    
    # KPI Metrics
    display_kpi_metrics(df)
    st.divider()

    # --- AGGREGATION (Calculate Early for Leaderboard) ---
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

    # # --- MASTER LEADERBOARD (Moved to Top) ---
    # st.header("📋 Master Leaderboard")
    # lb = agg_df.sort_values('Conversions', ascending=False).head(int(row_count_lb))
    # lb.insert(0, 'Rank', range(1, len(lb) + 1))
    
    # max_c = int(agg_df['Conversions'].max()) if not agg_df.empty else 100
    # max_s = int(agg_df['Normalized_Spend_USD'].max()) if not agg_df.empty else 100

    # st.dataframe(
    #    lb[['Rank', 'Ad ID', 'Ad name', 'Conversions', 'Normalized_Spend_USD', 'CPA', 'Facebook Video Link']],
    #    column_config={
    #        "Conversions": st.column_config.ProgressColumn("Convs", format="%d", min_value=0, max_value=max_c, color="green"),
    #        "Normalized_Spend_USD": st.column_config.ProgressColumn("Spend ($)", format="$%.2f", min_value=0, max_value=max_s, color="yellow"),
    #        "CPA": st.column_config.NumberColumn("CPA ($)", format="$%.2f"),
    #        "Facebook Video Link": st.column_config.LinkColumn("Ad", display_text="📺 View")
    #    },
    #    hide_index=True,
    #    width="stretch"
    # )
    # st.markdown(f"**Showing top {len(lb)} of {len(agg_df)} ads.**") 


    # --- MASTER LEADERBOARD (Pagination Logic) ---
    st.header("📋 Master Leaderboard", help=f"Ranking based on Conversion count")

    # 1. Setup Pagination Variables
    rows_per_page = int(row_count_lb) 
    total_rows = len(agg_df)
    num_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)

    # 2. Prepare Data (Sorted & Ranked)
    full_lb = agg_df.sort_values('Conversions', ascending=False).copy()
    full_lb.insert(0, 'Rank', range(1, len(full_lb) + 1))
    
    # Check if page is initialized in session state to prevent reset on interaction
    if "lb_page_selector" not in st.session_state:
        st.session_state.lb_page_selector = 1

    # 3. Calculate Slices
    start_idx = (st.session_state.lb_page_selector - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    lb_page = full_lb.iloc[start_idx:end_idx]
    
    max_c = int(agg_df['Conversions'].max()) if not agg_df.empty else 100
    max_s = int(agg_df['Normalized_Spend_USD'].max()) if not agg_df.empty else 100

    # 4. Display Table
    row_count_current = len(lb_page)
    # This logic says: "Show the actual number of rows, but stop growing at 50"
    dynamic_height = (min(row_count_current, 50) * 35) + 38 

    # 4. Display Table with Max-Height Logic
    st.dataframe(
        lb_page[['Rank', 'Ad ID', 'Ad name', 'Conversions', 'Normalized_Spend_USD', 'CPA', 'Facebook Video Link']],
        column_config={
            "Rank": st.column_config.NumberColumn("Rank"),
            "Conversions": st.column_config.ProgressColumn("Convs", format="%d", min_value=0, max_value=max_c, color="green"),
            "Normalized_Spend_USD": st.column_config.ProgressColumn("Spend ($)", format="$%.2f", min_value=0, max_value=max_s, color="yellow"),
            "CPA": st.column_config.NumberColumn("CPA ($)", format="$%.2f"),
            "Facebook Video Link": st.column_config.LinkColumn("Ad", display_text="📺 View")
        },
        hide_index=True,
        width="stretch",
        height=dynamic_height  # <--- Dynamic but capped at 50 rows
    )


    # 5. UPDATED FOOTER: Reset (Left) | Page Input (Middle) | Summary (Right)
    # [0.8, 0.6, 4] provides a small space for the reset button and input, leaving the rest for text
    foot_col1, foot_col2, foot_col3 = st.columns([0.8, 1, 4], vertical_alignment="center")

    with foot_col1:
        # "Back to Page 1" Button
        # It only shows as clickable if the user is NOT already on page 1
        if st.button("⏪ First Page", disabled=(st.session_state.lb_page_selector == 1)):
            st.session_state.lb_page_selector = 1
            st.rerun()

    with foot_col2:
        st.number_input(
            "Page", 
            min_value=1, max_value=num_pages, step=1, key="lb_page_selector", label_visibility="collapsed"
        )

    with foot_col3:
        st.markdown(f"**Page {st.session_state.lb_page_selector} of {num_pages}**. (Ads {start_idx + 1} to {min(end_idx, total_rows)} - of {total_rows})")


    if view_option == "Standard Dashboard":
        st.divider()

    # --- TRENDS ---
    # if view_option in ["Standard Dashboard", "Focus: Trends"]:
    #     st.subheader("📈 Conversion Trend")
    #     top_ids = df.groupby('Ad ID')['Conversions'].sum().nlargest(15).index
    #     trend_df = df[df['Ad ID'].isin(top_ids)].groupby(['Date', 'Ad name'])['Conversions'].sum().reset_index()
        
    #     h = 700 if "Focus" in view_option else 400
    #     fig = px.line(trend_df, x="Date", y="Conversions", color="Ad name", markers=True, template="plotly_dark", height=h)
    #     st.plotly_chart(fig, width="stretch")


    # --- TRENDS ---
    if view_option in ["Standard Dashboard", "Focus: Trends"]:
        st.subheader("📈 Trend Analysis")
        
        # 1. TUNE CONTROLS
        
        # CALCULATE DATE LOGIC HERE
        # Count how many days are in the selected range
        n_days = (end_date - start_date).days + 1
        
        # Set dynamic constraints:
        # Max smoothing is 15, OR the number of days available (whichever is smaller)
        dynamic_max = max(1, min(15, n_days))
        
        # Default value is 7, OR the dynamic_max (if range is smaller than 7 days)
        dynamic_value = min(7, dynamic_max)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            trend_metric = st.selectbox(
                "Select Metric", 
                options=['Conversions', 'Normalized_Spend_USD', 'CPA', 'Visits'], 
                format_func=lambda x: "Ad Spend ($)" if x == 'Normalized_Spend_USD' else x
            )
        with c2:
            # Slider for Moving Average (Smoothing)
            # Now uses dynamic_max and dynamic_value
            smoothing = st.slider(
                "Smoothing (Days)", 
                min_value=1, 
                max_value=dynamic_max, 
                value=dynamic_value, 
                help=f"Higher values smooth out daily spikes. Max limited to {dynamic_max} days based on your date range."
            )
        with c3:
            # Slider to limit number of lines
            top_n_trend = st.slider("Show Top N Ads", min_value=1, max_value=30, value=7)

        # 2. DATA PREPARATION
        if 'Total_Payout' not in df.columns:
            df['Total_Payout'] = df['CPA'] * df['Conversions']

        rank_metric = 'Conversions' if trend_metric == 'CPA' else trend_metric
        top_ids = df.groupby('Ad ID')[rank_metric].sum().nlargest(top_n_trend).index
        
        agg_cols = {
            'Conversions': 'sum',
            'Normalized_Spend_USD': 'sum',
            'Visits': 'sum',
            'Total_Payout': 'sum'
        }
        trend_data = df[df['Ad ID'].isin(top_ids)].groupby(['Date', 'Ad name']).agg(agg_cols).reset_index()
        
        if trend_metric == 'CPA':
            trend_data['CPA'] = trend_data['Total_Payout'] / trend_data['Conversions']
            trend_data['CPA'] = trend_data['CPA'].fillna(0).replace([np.inf, -np.inf], 0)
            y_col = 'CPA'
        else:
            y_col = trend_metric

        # 3. APPLY SMOOTHING
        if smoothing > 1:
            trend_data = trend_data.sort_values(['Ad name', 'Date'])
            trend_data[y_col] = trend_data.groupby('Ad name')[y_col].transform(lambda x: x.rolling(smoothing, min_periods=1).mean())

        # 4. PLOT CHART
        h = 700 if "Focus" in view_option else 500
        
        fig = px.line(
            trend_data, 
            x="Date", 
            y=y_col, 
            color="Ad name", 
            markers=True if smoothing == 1 else False, 
            template="plotly_dark", 
            height=h,
            render_mode="svg",
            title=f"Daily {trend_metric} {'(Smoothed ' + str(smoothing) + ' Days)' if smoothing > 1 else ''}"
        )
        
        fig.update_traces(
            line_shape='spline',
            line_width=2,
            marker_size=6,
            hovertemplate = '%{y:.2f}' if trend_metric in ['CPA', 'Normalized_Spend_USD'] else '%{y:.0f}'
        )
        
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


    # --- BREAKDOWN TABLES ---
    if view_option in ["Standard Dashboard", "Focus: Trends"]:
        tab_mo, tab_da = st.tabs(["📅 Monthly Breakdown", "🗓️ Daily Breakdown"])
        
        total_ads_in_view = df['Ad ID'].nunique()
        
        # Monthly Pivot
        with tab_mo:
            if not df.empty:
                p_mo = df.pivot_table(index=['Ad ID', 'Ad name', 'Facebook Video Link'], columns='Month Date', values='Conversions', aggfunc='sum', fill_value=0)
                p_mo.columns = [c.strftime('%b') for c in p_mo.columns]
                p_mo['Total'] = p_mo.sum(axis=1)
                
                # Sort and Limit by row_count_td
                p_mo = p_mo.sort_values('Total', ascending=False).head(int(row_count_td)).reset_index()
                
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
                st.markdown(f"**Showing top {len(p_mo)} of {total_ads_in_view} ads.**")

        # Daily Pivot
        with tab_da:
            if not df.empty:
                p_da = df.pivot_table(index=['Ad ID', 'Ad name', 'Facebook Video Link'], columns='Date', values='Conversions', aggfunc='sum', fill_value=0)
                p_da.columns = [c.strftime('%m/%d') for c in p_da.columns]
                p_da['Total'] = p_da.sum(axis=1)
                
                p_da = p_da.sort_values('Total', ascending=False).head(int(row_count_td))
                p_da.set_index('Total', append=True, inplace=True)
                
                st.dataframe(
                    p_da,
                    width="stretch",
                    column_config={"Facebook Video Link": st.column_config.LinkColumn("Link", display_text="View")}
                )
                st.markdown(f"**Showing top {len(p_da)} of {total_ads_in_view} ads.**")

    if view_option == "Standard Dashboard":
        st.divider()

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

if __name__ == "__main__":
    main()