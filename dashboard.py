import streamlit as st
import pandas as pd
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

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
    df['Month Date'] = pd.to_datetime(df['Month Date'], errors='coerce')
    df = df.dropna(subset=['Month Date']) 
    
    # 2. ID Sanitization
    df['Ad ID'] = df['Ad ID'].astype(str).str.strip().str.replace('.0', '', regex=False)
    
    # 3. Numeric Conversion
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 4. Percentage Scaling
    if df['Unique Visit %'].max() <= 1.0:
        df['Unique Visit %'] = df['Unique Visit %'] * 100

    # 5. Link Healing
    valid_links = df[df['Facebook Video Link'].str.contains('http', na=False, case=False)]
    ad_set_link_map = valid_links.groupby('Ad set name')['Facebook Video Link'].first().to_dict()
    
    def fill_missing_link(row):
        current_link = str(row['Facebook Video Link'])
        if 'http' not in current_link.lower():
            return ad_set_link_map.get(row['Ad set name'], "No Link Found")
        return current_link
    df['Facebook Video Link'] = df.apply(fill_missing_link, axis=1)
    
    # 6. Clickable Axis Labels
    def make_clickable(row):
        name = row['Ad name'] if pd.notna(row['Ad name']) else row['Ad ID']
        link = row.get('Facebook Video Link', '')
        if pd.notna(link) and 'http' in str(link):
            return f'<a href="{link}" target="_blank" style="color: #00CC96; text-decoration: none;">{name}</a>'
        return name
    df['Clickable_Label'] = df.apply(make_clickable, axis=1)
    return df

def display_kpi_metrics(df):
    """Display metrics using standard aggregation, excluding 0s from average."""
    t_spend = df['Normalized_Spend_USD'].sum()
    t_convs = df['Conversions'].sum()
    t_visits = df['Visits'].sum()
    t_u_visits = df['Unique visits'].sum()
    
    # Simple aggregation: Mean of the CPA column excluding 0s
    active_cpas = df[df['CPA'] > 0]['CPA']
    avg_cpa = active_cpas.mean() if not active_cpas.empty else 0
    
    u_pct = (t_u_visits / t_visits * 100) if t_visits > 0 else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Spend", f"${t_spend:,.0f}")
    c2.metric("Conversions", f"{t_convs:,.0f}")
    c3.metric("Avg CPA (Excl. 0s)", f"${avg_cpa:.2f}")
    c4.metric("Unique Visits", f"{t_u_visits:,.0f}")
    c5.metric("Unique %", f"{u_pct:.1f}%")

def main():
    raw_df = load_data()
    if raw_df is None:
        st.error("❌ Data file not found.")
        return
    
    # --- 1. INITIALIZE SESSION STATE ---
    if 'select_all' not in st.session_state:
        st.session_state.select_all = True
    if 'q1' not in st.session_state: st.session_state.q1 = False
    if 'q2' not in st.session_state: st.session_state.q2 = False
    if 'q3' not in st.session_state: st.session_state.q3 = False
    if 'q4' not in st.session_state: st.session_state.q4 = False

    # --- 2. CALLBACKS FOR MUTUAL EXCLUSION ---
    def on_all_change():
        if st.session_state.select_all:
            # If "Select All" is turned ON, turn OFF all individual quarters
            st.session_state.q1 = False
            st.session_state.q2 = False
            st.session_state.q3 = False
            st.session_state.q4 = False

    def on_q_change():
        # If any quarter is turned ON, turn OFF "Select All"
        st.session_state.select_all = False

    # --- 3. SIDEBAR CONTROLS ---
    st.sidebar.header("🕹️ Control Panel")
    view_option = st.sidebar.radio(
        "🔎 Dashboard View",
        options=["Standard Dashboard", "Focus: Monthly Trend", "Focus: Conversion Funnel", "Focus: Performance Chart"],
        index=0
    )
    
    st.sidebar.divider()
    
    # Get available months
    available_months = sorted(raw_df['Month Date'].dropna().unique())
    month_options = [d.strftime('%b %Y') for d in available_months]

    st.sidebar.write("📅 **Quick Filters**")
    
    # LINKED: Added 'key' and 'on_change' to link to the session state and logic
    st.sidebar.checkbox(
        "Select All Months", 
        key="select_all", 
        on_change=on_all_change
    )
    
    q_cols = st.sidebar.columns(2)
    q_cols[0].checkbox("Q1", key="q1", on_change=on_q_change)
    q_cols[1].checkbox("Q2", key="q2", on_change=on_q_change)
    q_cols[0].checkbox("Q3", key="q3", on_change=on_q_change)
    q_cols[1].checkbox("Q4", key="q4", on_change=on_q_change)

    # --- 4. DETERMINE SELECTION LIST ---
    # We now check against st.session_state directly
    if st.session_state.select_all:
        default_selection = month_options
    else:
        q_months = []
        for m_str in month_options:
            m_dt = datetime.strptime(m_str, '%b %Y')
            m_num = m_dt.month
            if st.session_state.q1 and m_num in [1, 2, 3]: q_months.append(m_str)
            if st.session_state.q2 and m_num in [4, 5, 6]: q_months.append(m_str)
            if st.session_state.q3 and m_num in [7, 8, 9]: q_months.append(m_str)
            if st.session_state.q4 and m_num in [10, 11, 12]: q_months.append(m_str)
        
        # Default to latest month if everything is unselected
        default_selection = q_months if q_months else [month_options[-1]]

    # Unique Key: Incorporates all session state values so the multiselect resets 
    # visually when any checkbox is clicked
    selector_key = f"ms_{st.session_state.select_all}_{st.session_state.q1}_{st.session_state.q2}_{st.session_state.q3}_{st.session_state.q4}"

    selected_months_str = st.sidebar.multiselect(
        "Select Months", 
        options=month_options, 
        default=default_selection,
        key=selector_key
    )

    selected_months_dt = [datetime.strptime(m, '%b %Y') for m in selected_months_str]
    
    # --- 5. SEARCH & GLOBAL FILTER ---
    search_q = st.sidebar.text_input("🔍 Search Ad Name/ID:")
    row_count = st.sidebar.number_input("🔢 Leaderboard Rows:", min_value=1, value=20)

    # Filter the main dataframe
    df = raw_df[raw_df['Month Date'].isin(selected_months_dt)]
    if search_q:
        df = df[df['Ad name'].str.contains(search_q, case=False) | df['Ad ID'].str.contains(search_q, case=False)]

    # --- HEADER & KPI METRICS ---
    st.title("🚀 2025 FB Ads Performance")
    display_kpi_metrics(df)
    st.divider()

    # --- TREND CHART ---
    if view_option in ["Standard Dashboard", "Focus: Monthly Trend"]:
        st.subheader("📈 Monthly Conversion Trend")
        top_14_ids = raw_df.groupby('Ad ID')['Conversions'].sum().nlargest(14).index
        trend_df = raw_df[raw_df['Ad ID'].isin(top_14_ids)].sort_values('Month Date')
        h = 700 if "Focus" in view_option else 400
        fig_trend = px.line(trend_df, x="Month Date", y="Conversions", color="Ad name", markers=True, template="plotly_dark", height=h)
        st.plotly_chart(fig_trend, use_container_width=True)
        #if "Focus" not in view_option: st.divider()

    # --- MoM COMPARISON TABLE ---
    if view_option in ["Standard Dashboard", "Focus: Monthly Trend"]:
        st.info("💡 Heatmap is calculated per row (excluding Total) to highlight individual performance trends across your selected months.")

        # Using 'df' ensures the sidebar Month Filtering and Search are applied to this table
        if not df.empty:
            # 1. Pivot the data: Ad ID/Name as rows, Month Date as columns
            pivot_df = df.pivot_table(
                index=['Ad ID', 'Ad name', 'Facebook Video Link'], 
                columns='Month Date', 
                values='Conversions', 
                aggfunc='sum',
                fill_value=0
            ).reset_index()

            # 2. Identify month columns and calculate Total
            month_cols_raw = [col for col in pivot_df.columns if isinstance(col, pd.Timestamp)]
            formatted_month_names = [col.strftime('%b %Y') for col in month_cols_raw]
            
            # Add Total Conversions for the selected sidebar range
            pivot_df['Total Conv'] = pivot_df[month_cols_raw].sum(axis=1)
            pivot_df = pivot_df.sort_values(by='Total Conv', ascending=False)

            # 3. Rename and Organize Columns: Ad ID, Name, Link (Hidden), Months, Total
            pivot_df.columns = ['Ad ID', 'Creative Name', 'Video_URL'] + formatted_month_names + ['Total Conv']
            
            # 4. Apply Heatmap Styling on a PER-ROW basis
            # We target only the monthly columns (excluding Ad ID, Name, and Total)
            styled_pivot = pivot_df.style.background_gradient(
                cmap='YlGn', 
                subset=formatted_month_names,
                axis=1  # Row-based scaling for maximum contrast per creative
            ).format({
                **{month: '{:,.0f}' for month in formatted_month_names},
                'Total Conv': '{:,.0f}'
            })

            # 5. Display the table with specific column configurations
            st.dataframe(
                styled_pivot,
                use_container_width=True,
                column_config={
                    "Ad ID": st.column_config.TextColumn(
                        "Ad ID", 
                        width=None, 
                        help="Full Facebook Ad ID"
                    ), 
                    "Creative Name": st.column_config.LinkColumn(
                        "Creative Name", 
                        width='small', # Auto-adjusts to the length of the name
                        help="Click to watch ad creative",
                    ),
                    "Video_URL": None, # Hides the raw link column
                    "Total Conv": st.column_config.NumberColumn(
                        "Total Conv", 
                        format="%d", 
                        help="Sum of conversions for selected sidebar months"
                    ),
                },
                hide_index=True
            )

            # Display Footer - count of ads
            current_count = len(pivot_df)
            total_count = raw_df['Ad ID'].nunique()
            st.markdown(f"📊 **Showing {current_count} of {total_count} ads**")
        else:
            st.warning("No data matches the current sidebar filters.")
    st.divider()

    # --- STANDARD DATA AGGREGATION ---
    # We group by Ad and take the mean of CPA, filtering out 0s within the group if necessary
    agg_df = df.groupby(['Ad ID', 'Ad name', 'Clickable_Label', 'Facebook Video Link']).agg({
        'Conversions': 'sum', 
        'Visits': 'sum', 
        'Unique visits': 'sum', 
        'Normalized_Spend_USD': 'sum',
        'CPA': lambda x: x[x > 0].mean() # Mean of CPA per ad group, excluding 0s
    }).reset_index().fillna(0)

    if view_option == "Standard Dashboard":
        col_a, col_b = st.columns(2)
    else:
        col_a = col_b = st.container()

    if view_option in ["Standard Dashboard", "Focus: Conversion Funnel"]:
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
        with col_b:
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
    
    # Calculate max values for dynamic bar scaling
    max_convs = int(agg_df['Conversions'].max()) if not agg_df.empty else 100
    max_spend = int(agg_df['Normalized_Spend_USD'].max()) if not agg_df.empty else 100

    st.dataframe(
        leaderboard[['Rank', 'Ad ID', 'Ad name', 'Conversions', 'Normalized_Spend_USD', 'CPA', 'Facebook Video Link']],
        column_config={
            "Conversions": st.column_config.ProgressColumn(
                "Convs", 
                format="%d", 
                min_value=0, 
                max_value=max_convs, 
                color="green"
            ),
            "Normalized_Spend_USD": st.column_config.ProgressColumn(
                "Ad Spend ($)", 
                format="$%.2f", 
                min_value=0, 
                max_value=max_spend, 
                color="yellow"
            ),
            "CPA": st.column_config.NumberColumn("CPA ($)", format="$%.2f"),
            "Facebook Video Link": st.column_config.LinkColumn("Watch", display_text="📺 View Ad")
        },
        hide_index=True, use_container_width=True
    )

    #Footer Count 
    current_rows = len(leaderboard)
    total_filtered = len(agg_df)
    st.markdown(f"📊 **Showing {current_rows} of {total_filtered} ads** matching your filters.")

if __name__ == "__main__":
    main()