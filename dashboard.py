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
    df = pd.read_csv(REPORT_FILE, dtype={'Ad ID': str})
    df = df.dropna(subset=['Ad ID'])
    
    # 1. Date Sanitization
    if 'Day' in df.columns:
        df['Date'] = pd.to_datetime(df['Day'], errors='coerce')
    else:
        df['Date'] = pd.to_datetime(df['Month Date'], errors='coerce')

    df['Month Date'] = pd.to_datetime(df['Month Date'], errors='coerce')
    df = df.dropna(subset=['Date']) 
    
    # 2. ID Sanitization
    df['Ad ID'] = df['Ad ID'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # 3. Numeric Conversion
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 4. Percentage Scaling
    if df['Unique Visit %'].max() <= 1.0:
        df['Unique Visit %'] = df['Unique Visit %'] * 100
    
    # 5. Link Healing (Optimized: Vectorized + Check)
    # If the report.py ran correctly, this is just a fallback.
    if 'Ad set name' in df.columns and 'Facebook Video Link' in df.columns:
        # Simple fill for 'No Link' if possible, using vectorization logic where feasible or map
        # Creating a map of known links
        mask_valid = df['Facebook Video Link'].str.contains('http', na=False, case=False)
        if mask_valid.any():
            ad_set_map = df[mask_valid].set_index('Ad set name')['Facebook Video Link'].to_dict()
            # Apply map only to missing rows
            mask_missing = ~mask_valid
            df.loc[mask_missing, 'Facebook Video Link'] = df.loc[mask_missing, 'Ad set name'].map(ad_set_map).fillna("No Link")

    # 6. Clickable Axis Labels (for Charts)
    df['Clickable_Label'] = np.where(
        df['Facebook Video Link'].str.contains('http', na=False),
        '<a href="' + df['Facebook Video Link'] + '" target="_blank" style="color: #00CC96; text-decoration: none;">' + df['Ad name'].fillna(df['Ad ID']) + '</a>',
        df['Ad name'].fillna(df['Ad ID'])
    )
    return df

def display_kpi_metrics(df):
    # Spending is Lifetime per Ad ID, so we deduplicate to avoid summing the same ad's spend 30 times for 30 days
    t_spend = df.drop_duplicates(subset=['Ad ID'])['Normalized_Spend_USD'].sum()
    t_convs = df['Conversions'].sum()
    t_visits = df['Visits'].sum()
    t_u_visits = df['Unique visits'].sum()
    
    active_cpas = df[df['CPA'] > 0]['CPA']
    avg_cpa = active_cpas.mean() if not active_cpas.empty else 0
    u_pct = (t_u_visits / t_visits * 100) if t_visits > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Spend (Lifetime)", f"${t_spend:,.0f}", help="Sum of lifetime spend for ads active in selection")
    c2.metric("Conversions", f"{t_convs:,.0f}")
    c3.metric("Avg CPA", f"${avg_cpa:.2f}")
    c4.metric("Unique Visits", f"{t_u_visits:,.0f}")
    c5.metric("Unique %", f"{u_pct:.1f}%")

def main():
    raw_df = load_data()
    if raw_df is None:
        st.error("❌ Data file not found. Please run the report generator.")
        return
    
    # --- 1. SIDEBAR CONTROLS ---
    st.sidebar.header("🕹️ Control Panel")
    view_option = st.sidebar.radio(
        "🔎 Dashboard View",
        options=["Standard Dashboard", "Focus: Trends", "Focus: Funnel", "Focus: Performance Chart"],
        index=0
    )
    
    st.sidebar.divider()

    # --- 2. DATE FILTERS ---
    st.sidebar.write("📅 **Select Custom Date Range**")

    # Fetch data boundaries from the loaded dataframe
    min_data_date = raw_df['Date'].min().date()
    max_data_date = raw_df['Date'].max().date()

    # Define the initial default range
    default_start = date(2025, 1, 1)
    default_end = date(2025, 12, 31)

    # Use a form to batch the input and prevent reruns while typing
    with st.sidebar.form("date_filter_form"):
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=min_data_date,
            max_value=max_data_date,
            format="YYYY/MM/DD" 
        )
        
        # This button acts as the 'Enter' key for your typed dates
        submit_date = st.form_submit_button("Apply Date Range")

    # Persistent state management
    if 'final_date_range' not in st.session_state:
        st.session_state.final_date_range = (default_start, default_end)

    if submit_date:
        # Check if the user selected a full range (start and end)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            st.session_state.final_date_range = date_range

    # Use df_dates for your filtering logic below
    df_dates = st.session_state.final_date_range


    # --- 3. GLOBAL FILTERS ---
    search_q = st.sidebar.text_input("🔍 Search Ad Name/ID:")
    row_count = st.sidebar.number_input("🔢 Leaderboard Rows:", min_value=1, value=20)

    # Apply Filters
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = raw_df[(raw_df['Date'].dt.date >= start_date) & (raw_df['Date'].dt.date <= end_date)]
    else:
        df = raw_df

    if search_q:
        df = df[df['Ad name'].str.contains(search_q, case=False) | df['Ad ID'].str.contains(search_q, case=False)]

    # --- HEADER & KPI ---
    st.title("🚀 2025 FB Ads Performance")
    display_kpi_metrics(df)
    st.divider()

    # --- TREND CHART ---
    if view_option in ["Standard Dashboard", "Focus: Trends"]:
        st.subheader("📈 Conversion Trend")
        top_14_ids = raw_df.groupby('Ad ID')['Conversions'].sum().nlargest(14).index
        
        # Optimize Trend Chart
        trend_agg = df[df['Ad ID'].isin(top_14_ids)].groupby(['Date', 'Ad name'])['Conversions'].sum().reset_index()
        h = 700 if "Focus" in view_option else 400
        
        fig_trend = px.line(trend_agg, x="Date", y="Conversions", color="Ad name", markers=True, template="plotly_dark", height=h)
        st.plotly_chart(fig_trend, use_container_width=True)

    # --- BREAKDOWN TABLES ---
    if view_option in ["Standard Dashboard", "Focus: Trends"]:
        tab_month, tab_day = st.tabs(["📅 Monthly Breakdown", "🗓️ Daily Breakdown"])
        
        # TAB 1: Monthly
        with tab_month:
            if not df.empty:
                pivot_mo = df.pivot_table(
                    index=['Ad ID', 'Ad name', 'Facebook Video Link'], 
                    columns='Month Date', 
                    values='Conversions', 
                    aggfunc='sum',
                    fill_value=0
                )
                pivot_mo.columns = [c.strftime('%b %Y') for c in pivot_mo.columns]
                mo_cols = list(pivot_mo.columns)
                pivot_mo['Total Conv'] = pivot_mo.sum(axis=1)
                pivot_mo = pivot_mo.sort_values(by='Total Conv', ascending=False).reset_index()

                #column ordering
                new_order = ['Ad ID', 'Ad name', 'Facebook Video Link', 'Total Conv'] + mo_cols
                pivot_mo = pivot_mo[new_order]
                
                st.dataframe(
                    pivot_mo.style.background_gradient(cmap='YlGn', subset=mo_cols, axis=1),
                    use_container_width=True,
                    column_config={
                        "Ad name": st.column_config.TextColumn("Ad Name", width="small"), # SHOW ACTUAL NAME
                        "Facebook Video Link": st.column_config.LinkColumn("Watch", display_text="📺 View"), # SHOW LINK BUTTON
                        "Total Conv": st.column_config.NumberColumn("Total Conv", format="%d")
                    },
                    hide_index=True
                )

        # TAB 2: Daily (Clean Table, No Heatmap)
        with tab_day:
            if not df.empty:
                # 1. Create the pivot table
                pivot_da = df.pivot_table(
                    index=['Ad ID', 'Ad name', 'Facebook Video Link'], 
                    columns='Date', 
                    values='Conversions', 
                    aggfunc='sum',
                    fill_value=0
                )
                
                # 2. Format date columns
                pivot_da.columns = [c.strftime('%m/%d') for c in pivot_da.columns]
                da_cols = list(pivot_da.columns)
                
                # 3. Calculate Total and sort
                pivot_da['Total Conv'] = pivot_da.sum(axis=1)
                pivot_da = pivot_da.sort_values(by='Total Conv', ascending=False)

                # 4. REARRANGE: Move 'Total Conv' into the index so it also freezes
                # Currently, index has: Ad ID, Ad name, Facebook Video Link
                # We add Total Conv to it.
                pivot_da.set_index('Total Conv', append=True, inplace=True)

                # 5. Display the dataframe
                # Streamlit will pin all index levels to the left
                st.dataframe(
                    pivot_da,
                    use_container_width=True,
                    column_config={
                        "Ad name": st.column_config.TextColumn("Ad Name", width="small"),
                        "Facebook Video Link": st.column_config.LinkColumn("Watch", display_text="📺 View"),
                        "Total Conv": st.column_config.NumberColumn("Total Conv", format="%d")
                    }
                    # Note: hide_index=True would unfreeze them, so we keep it False or omit it.
                )

                
    
    if view_option == "Standard Dashboard":
        st.divider()

    # --- STANDARD DATA AGGREGATION ---
    agg_df = df.groupby(['Ad ID', 'Ad name', 'Clickable_Label', 'Facebook Video Link']).agg({
        'Conversions': 'sum', 
        'Visits': 'sum', 
        'Unique visits': 'sum', 
        'Normalized_Spend_USD': 'max', 
        'CPA': lambda x: x[x > 0].mean() 
    }).reset_index().fillna(0)

    if view_option in ["Standard Dashboard", "Focus: Funnel"]:
        col_a, col_b = st.columns(2) if view_option == "Standard Dashboard" else (st.container(), st.container())
        
        with col_a:
            st.subheader("🔥 Integrated Conversion Funnel")
            top_funnel = agg_df.nlargest(14, 'Conversions').sort_values('Conversions', ascending=True)
            fig_f = go.Figure()
            fig_f.add_trace(go.Bar(y=top_funnel['Clickable_Label'], x=top_funnel['Visits'], name='Visits', orientation='h', marker_color='#636EFA'))
            fig_f.add_trace(go.Bar(y=top_funnel['Clickable_Label'], x=top_funnel['Unique visits'], name='Unique Visits', orientation='h', marker_color='#00CC96'))
            fig_f.add_trace(go.Bar(y=top_funnel['Clickable_Label'], x=top_funnel['Conversions'], name='Convs', orientation='h', marker_color='#EF553B'))
            h = 800 if "Focus" in view_option else CHART_HEIGHT
            fig_f.update_layout(barmode='overlay', template="plotly_dark", height=h, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_f, use_container_width=True)

    if view_option in ["Standard Dashboard", "Focus: Performance Chart"]:
        if view_option == "Standard Dashboard":
            container = col_b
        else:
            container = st.container()
        
        with container:
            st.subheader("🏆 Conversions vs CPA")
            top_perf = agg_df.nlargest(14, 'Conversions').sort_values('Conversions', ascending=True)
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(y=top_perf['Clickable_Label'], x=top_perf['Conversions'], name='Convs', orientation='h', marker_color='#00CC96'))
            fig_p.add_trace(go.Scatter(y=top_perf['Clickable_Label'], x=top_perf['CPA'], name='CPA ($)', mode='markers+lines', marker=dict(size=10, color='#FFA15A'), xaxis='x2'))
            h = 800 if "Focus" in view_option else CHART_HEIGHT
            fig_p.update_layout(template="plotly_dark", height=h, xaxis=dict(title="Conversions"), xaxis2=dict(title="CPA ($)", overlaying='x', side='top', showgrid=False), margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_p, use_container_width=True)

    if view_option == "Standard Dashboard":
        st.divider()

    # --- LEADERBOARD ---
    st.header("📋 Master Creative Performance Leaderboard")
    leaderboard = agg_df.sort_values('Conversions', ascending=False).head(int(row_count))
    leaderboard.insert(0, 'Rank', range(1, len(leaderboard) + 1))
    
    max_convs = int(agg_df['Conversions'].max()) if not agg_df.empty else 100
    max_spend = int(agg_df['Normalized_Spend_USD'].max()) if not agg_df.empty else 100

    st.dataframe(
        leaderboard[['Rank', 'Ad ID', 'Ad name', 'Conversions', 'Normalized_Spend_USD', 'CPA', 'Facebook Video Link']],
        column_config={
            "Conversions": st.column_config.ProgressColumn("Convs", format="%d", min_value=0, max_value=max_convs, color="green"),
            "Normalized_Spend_USD": st.column_config.ProgressColumn("Ad Spend ($)", format="$%.2f", min_value=0, max_value=max_spend, color="yellow"),
            "CPA": st.column_config.NumberColumn("CPA ($)", format="$%.2f"),
            "Facebook Video Link": st.column_config.LinkColumn("Watch", display_text="📺 View Ad")
        },
        hide_index=True, use_container_width=True
    )
    st.markdown(f"📊 **Showing {len(leaderboard)} of {len(agg_df)} ads** matching your filters.")

if __name__ == "__main__":
    main()