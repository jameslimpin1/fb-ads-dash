import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
NUMERIC_COLUMNS = ['Conversions', 'Normalized_Spend_USD', 'Visits', 'Unique visits', 'CR%', 'Unique Visit %', 'Calculated_CPA']
PERCENTAGE_COLUMNS = ['Unique Visit %', 'CR%']
CHART_HEIGHT = 650  # Unified height for perfect alignment

@st.cache_data
def load_data():
    """Load and clean the Facebook ads data with link fallback logic."""
    if not os.path.exists(REPORT_FILE):
        return None
    
    # Load and force Ad ID as string to preserve data integrity
    df = pd.read_csv(REPORT_FILE, dtype={'Ad ID': str})
    
    # Clean Ad ID column - remove null and whitespace-only entries
    df = df.dropna(subset=['Ad ID'])
    df['Ad ID'] = df['Ad ID'].astype(str).str.strip()
    df = df[df['Ad ID'] != '']
    
    # Convert numeric columns with proper error handling
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Normalize percentage columns to 0-100 scale
    for col in PERCENTAGE_COLUMNS:
        if col in df.columns and df[col].max() <= 1.0:
            df[col] = df[col] * 100
            
    # --- NEW: FALLBACK LINK LOGIC ---
    # 1. Create a mapping of Ad Set Name to the first valid video link found in that set
    valid_links = df[df['Facebook Video Link'].str.contains('http', na=False, case=False)]
    ad_set_link_map = valid_links.groupby('Ad set name')['Facebook Video Link'].first().to_dict()
    
    # 2. Function to fill missing links using the map
    def fill_missing_link(row):
        current_link = str(row['Facebook Video Link'])
        if 'http' not in current_link.lower():
            # Attempt to get link from the same ad set
            return ad_set_link_map.get(row['Ad set name'], row['Facebook Video Link'])
        return row['Facebook Video Link']

    df['Facebook Video Link'] = df.apply(fill_missing_link, axis=1)
    # --------------------------------
    
    # Calculate CPA if not present
    if 'Calculated_CPA' not in df.columns:
        df['Calculated_CPA'] = np.where(
            df['Conversions'] > 0,
            df['Normalized_Spend_USD'] / df['Conversions'],
            0
        )
    
    # Create clickable labels for charts
    def make_clickable_label(row):
        name = row['Ad name'] if pd.notna(row['Ad name']) else row['Ad ID']
        link = row.get('Facebook Video Link', '')
        if pd.notna(link) and 'http' in str(link):
            return f'<a href="{link}" target="_blank" style="color: #00CC96; text-decoration: none;">{name}</a>'
        return name
    
    df['Clickable_Label'] = df.apply(make_clickable_label, axis=1)
    
    # Filter out inactive test data
    df = df[(df['Conversions'] > 0) | (df['Normalized_Spend_USD'] > 0)]
    
    return df.reset_index(drop=True)


def display_kpi_metrics(df):
    """Display high-level account KPI metrics."""
    total_spend = df['Normalized_Spend_USD'].sum()
    total_convs = df['Conversions'].sum()
    total_visits = df['Visits'].sum()
    
    avg_cpa = total_spend / total_convs if total_convs > 0 else 0
    avg_cr = (total_convs / total_visits * 100) if total_visits > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spend", f"${total_spend:,.0f}")
    with col2:
        st.metric("Conversions", f"{total_convs:,.0f}")
    with col3:
        st.metric("Global Avg CPA", f"${avg_cpa:.2f}")
    with col4:
        st.metric("Global Avg CR%", f"{avg_cr:.2f}%")

def create_integrated_funnel(df, top_n=14):
    """Create integrated funnel showing Visits, Uniques, and Convs overlaid."""
    top_ads = df.nlargest(top_n, 'Conversions').sort_values('Conversions', ascending=True)
    if len(top_ads) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top_ads['Clickable_Label'], x=top_ads['Visits'], name='Visits', orientation='h', marker_color='#636EFA'))
    fig.add_trace(go.Bar(y=top_ads['Clickable_Label'], x=top_ads['Unique visits'], name='Unique Visits', orientation='h', marker_color='#00CC96'))
    fig.add_trace(go.Bar(y=top_ads['Clickable_Label'], x=top_ads['Conversions'], name='Conversions', orientation='h', marker_color='#EF553B'))
    
    fig.update_layout(
        barmode='overlay', template="plotly_dark", height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=80, b=40), # Balanced top margin for secondary axis alignment
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Volume Count"
    )
    fig.update_yaxes(tickmode='array', tickvals=top_ads['Clickable_Label'], title_text="")
    return fig

def create_performance_chart(df, top_n=14):
    """Dual-axis chart showing Conversions vs CPA with clickable labels."""
    top_ads = df.nlargest(top_n, 'Conversions').sort_values('Conversions', ascending=True)
    if len(top_ads) == 0:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=top_ads['Clickable_Label'], x=top_ads['Conversions'], name='Conversions', orientation='h', marker_color='#00CC96', xaxis='x'))
    fig.add_trace(go.Scatter(y=top_ads['Clickable_Label'], x=top_ads['Calculated_CPA'], name='CPA ($)', mode='markers+lines', marker=dict(size=10, color='#FFA15A'), xaxis='x2'))
    
    fig.update_layout(
        template="plotly_dark", height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=80, b=40), # Aligned with Funnel chart margin
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Conversions", side='bottom'),
        xaxis2=dict(title="CPA ($)", overlaying='x', side='top', showgrid=False, tickprefix='$')
    )
    fig.update_yaxes(tickmode='array', tickvals=top_ads['Clickable_Label'], title_text="")
    return fig

def main():
    df = load_data()
    if df is None or len(df) == 0:
        st.error("❌ Data not found or empty.")
        return

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🛠️ Controls")
    search_q = st.sidebar.text_input("🔍 Search Ads:", placeholder="Ad ID or Name...")
    row_count = st.sidebar.number_input("🔢 Leaderboard Rows:", min_value=1, value=min(20, len(df)))

    st.title("🚀 2025 FB Ads Performance")
    display_kpi_metrics(df)
    
    st.divider()
    
    # Top Visuals - Aligned Horizontally
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔥 Integrated Conversion Funnel")
        funnel = create_integrated_funnel(df, top_n=14)
        if funnel: st.plotly_chart(funnel, use_container_width=True)
    with col_b:
        st.subheader("🏆 Conversions vs CPA")
        perf = create_performance_chart(df, top_n=14)
        if perf: st.plotly_chart(perf, use_container_width=True)
    
    st.divider()

    # --- MASTER LEADERBOARD ---
    st.header("📋 Master Creative Performance Leaderboard")
    st.info("💡 **Annotation:** Rank based on Conversion numbers")
    
    display_df = df.copy()
    if search_q:
        display_df = display_df[display_df['Ad name'].str.contains(search_q, case=False) | display_df['Ad ID'].str.contains(search_q, case=False)]
    
    total_available_rows = len(display_df)
    display_df = display_df.sort_values('Conversions', ascending=False).head(int(row_count))
    display_df.insert(0, 'Rank', range(1, len(display_df) + 1))

    # Dynamic height calculation to prevent empty gray rows
    dynamic_height = min(len(display_df) * 36 + 40, 800)

    st.dataframe(
        display_df[['Rank', 'Ad ID', 'Ad name', 'Conversions', 'Normalized_Spend_USD', 'Calculated_CPA', 'CR%', 'Unique Visit %', 'Facebook Video Link']],
        column_config={
            "Conversions": st.column_config.ProgressColumn("Conversions", format="%d", min_value=0, max_value=int(df['Conversions'].max()), color="green"),
            "Normalized_Spend_USD": st.column_config.ProgressColumn("Ad Spend ($)", format="$%.2f", min_value=0, max_value=int(df['Normalized_Spend_USD'].max()), color="yellow"),
            "Calculated_CPA": st.column_config.NumberColumn("CPA ($)", format="$%.2f"),
            "CR%": st.column_config.NumberColumn("CR %", format="%.2f%%"),
            "Unique Visit %": st.column_config.NumberColumn("Unique Visit %", format="%.2f%%"),
            "Facebook Video Link": st.column_config.LinkColumn("Watch Creative", display_text="📺 View Ad")
        },
        hide_index=True, use_container_width=True, height=dynamic_height
    )
    
    st.write(f"(showing {len(display_df)} of {total_available_rows} total rows)")

    # --- EXPORT SECTION ---
    st.divider()
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current View as CSV",
        data=csv_data,
        file_name=f"FB_Performance_Export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv',
    )

if __name__ == "__main__":
    main()