"""
🎨 VISUALIZATION - AI-Powered Dashboard Generator
Integrates with backend: smart_recommender.py + beautiful_dashboard.py logic
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import sys
from pathlib import Path

# Add visualization folder to path
viz_path = Path(__file__).parent.parent / "visualization"
sys.path.insert(0, str(viz_path))

from visualization.smart_recommender import SmartRecommender

# Page config
st.set_page_config(
    page_title="Visualization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Matching Data Cleaning/ML Training dark theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    /* Main theme - Dark futuristic with neon accents */
    .main {
        background: #0a0e27;
        background-image: 
            radial-gradient(at 20% 30%, rgba(13, 110, 253, 0.15) 0px, transparent 50%),
            radial-gradient(at 80% 70%, rgba(132, 59, 206, 0.15) 0px, transparent 50%),
            radial-gradient(at 40% 90%, rgba(13, 202, 240, 0.1) 0px, transparent 50%);
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
    }
    
    /* Block container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers - Neon glow effect */
    h1 {
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(13, 202, 240, 0.8);
        letter-spacing: 2px;
    }
    
    h2 {
        color: #0dcaf0 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(13, 202, 240, 0.6);
    }
    
    h3 {
        color: #6610f2 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(102, 16, 242, 0.5);
    }
    
    /* All text white */
    .stMarkdown, p, span, label {
        color: #ffffff !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(13, 110, 253, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* KPI Cards - Neon glow style */
    .kpi-card {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.2) 0%, rgba(102, 16, 242, 0.2) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(13, 110, 253, 0.5);
        color: white;
        text-align: center;
        box-shadow: 0 0 30px rgba(13, 110, 253, 0.3);
        transition: all 0.3s;
        backdrop-filter: blur(10px);
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(13, 110, 253, 0.5);
        border-color: rgba(13, 110, 253, 0.8);
    }
    
    .kpi-card-green {
        background: linear-gradient(135deg, rgba(32, 201, 151, 0.2) 0%, rgba(13, 202, 240, 0.2) 100%);
        border: 2px solid rgba(32, 201, 151, 0.5);
        box-shadow: 0 0 30px rgba(32, 201, 151, 0.3);
    }
    
    .kpi-card-green:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(32, 201, 151, 0.5);
        border-color: rgba(32, 201, 151, 0.8);
    }
    
    .kpi-card-orange {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(214, 51, 132, 0.2) 100%);
        border: 2px solid rgba(255, 107, 107, 0.5);
        box-shadow: 0 0 30px rgba(255, 107, 107, 0.3);
    }
    
    .kpi-card-orange:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(255, 107, 107, 0.5);
        border-color: rgba(255, 107, 107, 0.8);
    }
    
    .kpi-card-blue {
        background: linear-gradient(135deg, rgba(13, 202, 240, 0.2) 0%, rgba(102, 126, 234, 0.2) 100%);
        border: 2px solid rgba(13, 202, 240, 0.5);
        box-shadow: 0 0 30px rgba(13, 202, 240, 0.3);
    }
    
    .kpi-card-blue:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 40px rgba(13, 202, 240, 0.5);
        border-color: rgba(13, 202, 240, 0.8);
    }
    
    .kpi-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .kpi-label {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* AI Recommendation Box - Neon style */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 107, 107, 0.2) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid rgba(255, 193, 7, 0.5);
        border-left: 5px solid #ffc107;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 0 30px rgba(255, 193, 7, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .recommendation-box h4 {
        color: #ffc107 !important;
        margin-top: 0;
        font-weight: 700;
        font-size: 1.3rem;
        text-shadow: 0 0 10px rgba(255, 193, 7, 0.5);
    }
    
    .recommendation-box p {
        color: white !important;
        margin-bottom: 0;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.2) 0%, rgba(102, 16, 242, 0.2) 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid rgba(13, 110, 253, 0.5);
        color: white;
        box-shadow: 0 0 20px rgba(13, 110, 253, 0.3);
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(214, 51, 132, 0.2) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid rgba(255, 107, 107, 0.5);
        color: white;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 107, 107, 0.3);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(13, 202, 240, 0.6), transparent);
        box-shadow: 0 0 10px rgba(13, 202, 240, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(10, 14, 39, 0.5);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(13, 110, 253, 0.2);
        border: 1px solid rgba(13, 110, 253, 0.3);
        border-radius: 8px;
        color: white;
        font-family: 'Orbitron', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.4) 0%, rgba(102, 16, 242, 0.4) 100%);
        border-color: rgba(13, 202, 240, 0.8);
        box-shadow: 0 0 20px rgba(13, 202, 240, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.3) 0%, rgba(102, 16, 242, 0.3) 100%);
        border: 2px solid rgba(13, 110, 253, 0.5);
        color: white;
        font-family: 'Orbitron', sans-serif;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(13, 110, 253, 0.3);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        border-color: rgba(13, 202, 240, 0.8);
        box-shadow: 0 0 30px rgba(13, 202, 240, 0.5);
        transform: translateY(-2px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.2) 0%, rgba(102, 16, 242, 0.2) 100%);
        border: 1px solid rgba(13, 110, 253, 0.3);
        border-radius: 10px;
        color: white !important;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: white !important;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Plotly charts background */
    .js-plotly-plot {
        background-color: rgba(10, 14, 39, 0.5) !important;
        border-radius: 10px;
        border: 1px solid rgba(13, 110, 253, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'viz_recommender' not in st.session_state:
    st.session_state.viz_recommender = None
if 'column_groups' not in st.session_state:
    st.session_state.column_groups = []
if 'selected_viz_columns' not in st.session_state:
    st.session_state.selected_viz_columns = []
if 'viz_recommendations' not in st.session_state:
    st.session_state.viz_recommendations = []
if 'dashboard_generated' not in st.session_state:
    st.session_state.dashboard_generated = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = None

# Header
st.title("🎨 AI-Powered Visualization")
st.markdown("**PowerBI-Style Dashboard Generator with Smart Learning**")

# Get data from session state - Check multiple sources (like ML page)
def get_viz_data():
    """Get data for visualization - check multiple sources"""
    if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df, "cleaned"
    elif 'current_df' in st.session_state and st.session_state.current_df is not None:
        return st.session_state.current_df, "current"
    elif 'original_df' in st.session_state and st.session_state.original_df is not None:
        return st.session_state.original_df, "original"
    elif 'raw_data' in st.session_state and st.session_state.raw_data is not None:
        return st.session_state.raw_data, "raw"
    elif 'df' in st.session_state and st.session_state.df is not None:
        return st.session_state.df, "df"
    return None, None

viz_data, data_source = get_viz_data()

# Sidebar - Matching beautiful_dashboard.py style
with st.sidebar:
    st.markdown("### ⚙️ VISUALIZATION CONFIG")
    
    if st.session_state.viz_recommendations:
        st.metric("Column Groups", len(st.session_state.column_groups))
        st.metric("Total Columns", len(st.session_state.selected_viz_columns))
    
    st.markdown("---")
    st.markdown("### 📊 LIVE METRICS")
    
    if viz_data is not None:
        st.metric("📥 Dataset Rows", f"{len(viz_data):,}")
        st.metric("📋 Columns", len(viz_data.columns))
        st.info(f"📂 Data Source: {data_source}")
        
        if st.session_state.viz_recommender:
            db_stats = st.session_state.viz_recommender.db.get_database_stats()
            st.metric("🧠 Learned Patterns", db_stats['learned_patterns'])
            st.metric("💾 DB Recommendations", db_stats['database_recommendations'])
            st.metric("🤖 LLM Calls", db_stats['llm_recommendations'])
            
            if db_stats['total_recommendations'] > 0:
                db_pct = (db_stats['database_recommendations'] / db_stats['total_recommendations']) * 100
                st.metric("⚡ Efficiency", f"{db_pct:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🎯 SYSTEM STATUS")
    st.markdown("""
    <div class="info-box" style="font-size: 0.85rem; padding: 1rem;">
        <strong>🎨 Visualization Engine</strong><br><br>
        ⚡ Smart recommendations<br>
        🧠 Learning database<br>
        📊 PowerBI-style charts<br>
        🎛️ Interactive dashboards
    </div>
    """, unsafe_allow_html=True)

# Main content
if viz_data is None:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ NO DATA AVAILABLE</h3>
        <p>Please upload and process data first.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Create tabs - matching CLI workflow
    tab1, tab2, tab3 = st.tabs([
        "🎯 Column Selection",
        "🤖 AI Recommendations",
        "📊 Dashboard"
    ])
    
    # TAB 1: Column Selection (from visualization_cli_powerbi.py)
    with tab1:
        st.markdown("## 🎯 Select Columns for Visualization")
        st.info(f"📊 Available: {len(viz_data.columns)} columns × {len(viz_data):,} rows")
        
        # Show available columns
        st.markdown("### 📋 Available Columns")
        cols_display = st.columns(4)
        for idx, col in enumerate(viz_data.columns):
            with cols_display[idx % 4]:
                dtype_emoji = "🔢" if pd.api.types.is_numeric_dtype(viz_data[col]) else "📝"
                st.markdown(f"{dtype_emoji} **{col}**")
                st.caption(f"({viz_data[col].dtype})")
        
        st.markdown("---")
        
        # Multi-group selection (matching CLI multi-group support)
        st.markdown("### ✅ Create Column Groups")
        selected_cols = st.multiselect(
            "Select columns for this group:",
            options=list(viz_data.columns),
            key="col_selector"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("✅ Add Group", type="primary", use_container_width=True):
                if selected_cols:
                    st.session_state.column_groups.append(selected_cols)
                    for col in selected_cols:
                        if col not in st.session_state.selected_viz_columns:
                            st.session_state.selected_viz_columns.append(col)
                    st.success(f"✅ Added group {len(st.session_state.column_groups)}!")
                    st.rerun()
        
        with col_b:
            if st.button("🗑️ Clear All Groups", use_container_width=True):
                st.session_state.column_groups = []
                st.session_state.selected_viz_columns = []
                st.session_state.viz_recommendations = []
                st.session_state.dashboard_generated = False
                st.rerun()
        
        # Display current groups
        if st.session_state.column_groups:
            st.markdown("---")
            st.markdown("### 📊 Current Groups")
            for i, group in enumerate(st.session_state.column_groups, 1):
                st.markdown(f"**Group {i}:** {', '.join(group)}")
            st.success(f"✅ Total: {len(st.session_state.column_groups)} group(s)")
    
    # TAB 2: AI Recommendations (using backend SmartRecommender)
    with tab2:
        st.markdown("## 🤖 AI-Powered Recommendations")
        st.markdown("**Smart System: Database-first (fast & free) → LLM fallback (learns)**")
        
        if not st.session_state.column_groups:
            st.warning("⚠️ Please add column groups in Tab 1 first")
        else:
            # Initialize recommender
            if st.session_state.viz_recommender is None:
                if st.button("🚀 Initialize AI Recommender", type="primary", use_container_width=True):
                    with st.spinner("Initializing Smart Recommender..."):
                        try:
                            st.session_state.viz_recommender = SmartRecommender(confidence_threshold=0.75)
                            db_stats = st.session_state.viz_recommender.db.get_database_stats()
                            st.success(f"✅ Ready! {db_stats['total_sessions']} sessions in database")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            # Show recommender status
            if st.session_state.viz_recommender:
                st.success("✅ Smart Recommender Active")
                
                # Get recommendations (matching CLI workflow EXACTLY)
                if st.button("🧠 Get AI Recommendations", type="primary", use_container_width=True):
                    with st.spinner("Analyzing data..."):
                        try:
                            # Record session (matching CLI line 104)
                            # Get dataset name from uploaded file or use generic name
                            if hasattr(st.session_state.get('uploaded_file'), 'name'):
                                dataset_name = st.session_state.uploaded_file.name
                            else:
                                dataset_name = f'dataset_{data_source}.csv'
                            
                            session_id = st.session_state.viz_recommender.db.record_session(
                                dataset_name=dataset_name,
                                df=viz_data,
                                selected_columns=st.session_state.selected_viz_columns,
                                group_structure=st.session_state.column_groups
                            )
                            st.session_state.session_id = session_id
                            
                            # Get recommendations for each group (matching CLI lines 116-141)
                            all_recommendations = []
                            for i, group in enumerate(st.session_state.column_groups, 1):
                                st.info(f"📊 Analyzing Group {i}: {', '.join(group)}")
                                
                                # USE SMART RECOMMENDER (matching CLI line 120)
                                result = st.session_state.viz_recommender.get_recommendations(
                                    df=viz_data,
                                    selected_columns=group,
                                    dataset_name=dataset_name,
                                    session_id=session_id
                                )
                                
                                # Show results (matching CLI lines 124-133)
                                if result['source'] == 'database':
                                    st.success(f"✅ Group {i}: Database (confidence: {result['confidence']*100:.0f}%) - $0.00")
                                elif result['source'] == 'llm':
                                    st.info(f"🤖 Group {i}: LLM ({result['reasoning']}) - ${result.get('cost', 0):.2f}")
                                else:
                                    st.info(f"🔄 Group {i}: Hybrid (confidence: {result['confidence']*100:.0f}%)")
                                
                                all_recommendations.append({
                                    'group_id': i,
                                    'columns': group,
                                    'recommendations': result['recommendations']
                                })
                            
                            st.session_state.viz_recommendations = all_recommendations
                            st.session_state.dashboard_generated = True
                            
                            # Show efficiency stats (matching CLI lines 145-150)
                            session_stats = st.session_state.viz_recommender.get_stats()
                            if session_stats['db_hits'] > 0:
                                st.success(f"""
                                💡 Session Efficiency:
                                - 💾 Database: {session_stats['db_hits']} time(s)
                                - 🤖 LLM: {session_stats['llm_calls']} time(s)
                                - 💰 Saved: ${session_stats.get('estimated_savings', 0):.2f}
                                """)
                            
                            st.success("✅ Ready! Go to Dashboard tab")
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Display current recommendations
                if st.session_state.viz_recommendations:
                    st.markdown("---")
                    st.markdown("### 📊 Current Recommendations")
                    for group_data in st.session_state.viz_recommendations:
                        with st.expander(f"📊 Group {group_data['group_id']}: {', '.join(group_data['columns'])}", expanded=True):
                            recs = group_data['recommendations']
                            if 'overall_recommendation' in recs:
                                st.markdown(f"""
                                <div class="recommendation-box">
                                    <h4>🤖 Overall Recommendation</h4>
                                    <p>{recs['overall_recommendation']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            if 'individual_visualizations' in recs:
                                st.markdown("**Visualizations:**")
                                for viz in recs['individual_visualizations']:
                                    st.markdown(f"- **{viz.get('column', 'N/A')}** → {viz.get('viz_type', 'N/A')}")
    
    # TAB 3: Dashboard (using beautiful_dashboard.py logic EXACTLY)
    with tab3:
        if not st.session_state.dashboard_generated:
            st.warning("⚠️ Please get recommendations in Tab 2 first")
        else:
            # Get unique columns
            all_selected_cols = []
            seen_cols = set()
            for group in st.session_state.column_groups:
                for col in group:
                    if col not in seen_cols:
                        all_selected_cols.append(col)
                        seen_cols.add(col)
            
            df_selected = viz_data[all_selected_cols]
            
            # Dashboard page selector (matching beautiful_dashboard.py line 262)
            st.markdown("### 🎯 Dashboard Pages")
            page = st.radio("", ["📊 Statistics Report", "🎛️ PowerBI Dashboard"], label_visibility="collapsed")
            st.markdown("---")
            
            # ==================== PAGE 1: STATISTICS REPORT ====================
            if page == "📊 Statistics Report":
                st.title("📊 Advanced Statistics Report")
                st.markdown("**AI-Powered Data Analysis with Beautiful Visualizations**")
                st.markdown("---")
                
                if len(st.session_state.column_groups) > 1:
                    st.info(f"📊 Analyzing {len(st.session_state.column_groups)} Column Groups")
                
                # Process each group (matching beautiful_dashboard.py lines 278-420)
                for group_data in st.session_state.viz_recommendations:
                    group_id = group_data['group_id']
                    group_cols = group_data['columns']
                    recommendations = group_data['recommendations']
                    
                    with st.expander(f"📊 GROUP {group_id}: {', '.join(group_cols)}", expanded=(len(st.session_state.column_groups) == 1)):
                        # AI Recommendation
                        if recommendations and 'overall_recommendation' in recommendations:
                            st.markdown(f"""
                            <div class="recommendation-box">
                                <h4>🤖 AI Recommendation</h4>
                                <p>{recommendations['overall_recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Summary Statistics
                        st.subheader("📈 Summary Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Descriptive Statistics**")
                            st.dataframe(viz_data[group_cols].describe().round(2), use_container_width=True)
                        
                        with col2:
                            st.markdown("**📉 Additional Metrics**")
                            metrics_df = pd.DataFrame({
                                'Metric': ['Count', 'Missing', 'Unique', 'Mode'],
                                **{col: [
                                    viz_data[col].count(),
                                    viz_data[col].isnull().sum(),
                                    viz_data[col].nunique(),
                                    viz_data[col].mode()[0] if len(viz_data[col].mode()) > 0 else 'N/A'
                                ] for col in group_cols}
                            })
                            st.dataframe(metrics_df, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Correlation Matrix
                        numeric_cols = [col for col in group_cols if pd.api.types.is_numeric_dtype(viz_data[col])]
                        if len(numeric_cols) >= 2:
                            st.subheader("🔗 Correlation Matrix")
                            corr_matrix = viz_data[numeric_cols].corr()
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                colorscale='RdBu_r',
                                zmid=0,
                                text=corr_matrix.values.round(2),
                                texttemplate='%{text}',
                                textfont={"size": 12},
                                colorbar=dict(title="Correlation")
                            ))
                            fig.update_layout(title="Correlation Heatmap", height=400)
                            st.plotly_chart(fig, use_container_width=True, key=f"corr_{group_id}")
                            st.markdown("---")
                        
                        # Individual Analysis with 2x2 grid (matching beautiful_dashboard.py lines 350-390)
                        st.subheader("📊 Individual Column Analysis")
                        if recommendations and 'individual_visualizations' in recommendations:
                            for viz in recommendations['individual_visualizations']:
                                col_name = viz['column']
                                if col_name not in viz_data.columns or not pd.api.types.is_numeric_dtype(viz_data[col_name]):
                                    continue
                                
                                st.markdown(f"### 📌 {col_name}")
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # 2x2 grid (EXACT match to beautiful_dashboard.py)
                                    fig = make_subplots(
                                        rows=2, cols=2,
                                        subplot_titles=('Distribution', 'Box Plot', 'Violin Plot', 'Density'),
                                        specs=[[{"type": "histogram"}, {"type": "box"}],
                                               [{"type": "violin"}, {"type": "scatter"}]]
                                    )
                                    
                                    fig.add_trace(go.Histogram(x=viz_data[col_name], marker_color='#667eea'), row=1, col=1)
                                    fig.add_trace(go.Box(y=viz_data[col_name], marker_color='#11998e'), row=1, col=2)
                                    fig.add_trace(go.Violin(y=viz_data[col_name], fillcolor='#f093fb'), row=2, col=1)
                                    
                                    hist, bins = np.histogram(viz_data[col_name].dropna(), bins=30, density=True)
                                    bin_centers = (bins[:-1] + bins[1:]) / 2
                                    fig.add_trace(go.Scatter(x=bin_centers, y=hist, mode='lines', fill='tozeroy',
                                                           line=dict(color='#4facfe', width=3)), row=2, col=2)
                                    
                                    fig.update_layout(height=600, showlegend=False, title_text=f"Analysis: {col_name}")
                                    st.plotly_chart(fig, use_container_width=True, key=f"ind_{group_id}_{col_name}")
                                
                                with col2:
                                    st.markdown("**🤖 AI Insights**")
                                    if 'reasoning' in viz:
                                        st.info(viz['reasoning'])
                                    st.markdown("**📊 Key Statistics**")
                                    st.metric("Mean", f"{viz_data[col_name].mean():.2f}")
                                    st.metric("Median", f"{viz_data[col_name].median():.2f}")
                                    st.metric("Std Dev", f"{viz_data[col_name].std():.2f}")
                                
                                st.markdown("---")
                        
                        # Relationship Analysis (matching beautiful_dashboard.py lines 404-480)
                        if recommendations and 'relationship_visualizations' in recommendations:
                            st.subheader("🔗 Relationship Analysis")
                            for viz in recommendations['relationship_visualizations']:
                                cols = viz['columns']
                                if not all(col in viz_data.columns and pd.api.types.is_numeric_dtype(viz_data[col]) for col in cols):
                                    continue
                                
                                st.markdown(f"### 🔗 {cols[0]} vs {cols[1]}")
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    fig = make_subplots(
                                        rows=2, cols=2,
                                        subplot_titles=('Scatter', '2D Density', 'Hexbin', 'Joint'),
                                        specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                               [{"type": "histogram2d"}, {"type": "scatter"}]]
                                    )
                                    
                                    fig.add_trace(go.Scatter(x=viz_data[cols[0]], y=viz_data[cols[1]], mode='markers',
                                                           marker=dict(color='#667eea', size=8, opacity=0.6)), row=1, col=1)
                                    
                                    z = np.polyfit(viz_data[cols[0]], viz_data[cols[1]], 1)
                                    p = np.poly1d(z)
                                    fig.add_trace(go.Scatter(x=viz_data[cols[0]], y=viz_data[cols[1]], mode='markers',
                                                           marker=dict(color='#11998e', size=8, opacity=0.6)), row=1, col=2)
                                    fig.add_trace(go.Scatter(x=sorted(viz_data[cols[0]]), y=p(sorted(viz_data[cols[0]])),
                                                           mode='lines', line=dict(color='red', width=3)), row=1, col=2)
                                    
                                    fig.add_trace(go.Histogram2d(x=viz_data[cols[0]], y=viz_data[cols[1]], colorscale='Blues'), row=2, col=1)
                                    fig.add_trace(go.Scatter(x=viz_data[cols[0]], y=viz_data[cols[1]], mode='markers',
                                                           marker=dict(color=viz_data[cols[1]], colorscale='Viridis', size=10, showscale=True)), row=2, col=2)
                                    
                                    fig.update_layout(height=700, showlegend=False, title_text=f"{cols[0]} vs {cols[1]}")
                                    st.plotly_chart(fig, use_container_width=True, key=f"rel_{group_id}_{cols[0]}_{cols[1]}")
                                
                                with col2:
                                    st.markdown("**🤖 AI Insights**")
                                    if 'reasoning' in viz:
                                        st.info(viz['reasoning'])
                                    st.markdown("**📈 Correlation**")
                                    corr = viz_data[cols[0]].corr(viz_data[cols[1]])
                                    st.metric("Correlation", f"{corr:.4f}")
                                    if abs(corr) > 0.7:
                                        st.success("🟢 Strong")
                                    elif abs(corr) > 0.4:
                                        st.warning("🟡 Moderate")
                                    else:
                                        st.error("🔴 Weak")
                                
                                st.markdown("---")
            
            # ==================== PAGE 2: POWERBI DASHBOARD ====================
            else:
                st.title("🎛️ PowerBI-Style Dashboard")
                st.markdown("**Professional Business Intelligence Visualizations**")
                st.markdown("---")
                
                if len(st.session_state.column_groups) > 1:
                    st.info(f"📊 Displaying {len(all_selected_cols)} columns from {len(st.session_state.column_groups)} groups")
                
                # KPI Cards (matching beautiful_dashboard.py lines 553-579)
                numeric_cols = [col for col in all_selected_cols if pd.api.types.is_numeric_dtype(viz_data[col])]
                if numeric_cols:
                    st.subheader("📈 Key Performance Indicators")
                    cols_display = st.columns(min(4, len(numeric_cols)))
                    colors = ['kpi-card', 'kpi-card-green', 'kpi-card-orange', 'kpi-card-blue']
                    
                    for i, col in enumerate(numeric_cols[:4]):
                        with cols_display[i]:
                            avg_val = viz_data[col].mean()
                            display_val = f"{avg_val:,.2f}" if avg_val < 1000 else f"{avg_val:,.0f}"
                            st.markdown(f"""
                            <div class="{colors[i % 4]}">
                                <div class="kpi-value">{display_val}</div>
                                <div class="kpi-label">Average {col}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("---")
                
                # PowerBI Visualizations (matching beautiful_dashboard.py EXACTLY)
                st.subheader("📊 PowerBI Visualizations")
                
                # Row 1: Donut + 3D Bar (lines 586-654)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🍩 Anneau (Donut)")
                    # Find categorical columns (more robust detection)
                    cat_cols = []
                    for col in all_selected_cols:
                        if viz_data[col].dtype == 'object':
                            cat_cols.append(col)
                        elif pd.api.types.is_numeric_dtype(viz_data[col]) and viz_data[col].nunique() < 20:
                            cat_cols.append(col)
                    
                    if cat_cols:
                        cat_col = cat_cols[0]
                        vc = viz_data[cat_col].value_counts().head(10).reset_index()  # Top 10 only
                        vc.columns = [cat_col, 'count']
                        
                        if len(vc) > 0:  # Make sure we have data
                            fig = go.Figure(data=[go.Pie(
                                labels=vc[cat_col],
                                values=vc['count'],
                                hole=0.5,
                                marker=dict(
                                    colors=px.colors.qualitative.Set3,
                                    line=dict(color='rgba(10, 14, 39, 0.8)', width=2)
                                ),
                                textinfo='label+percent',
                                textposition='outside',
                                textfont=dict(size=12, color='white'),
                                pull=[0.1 if i == 0 else 0 for i in range(len(vc))]
                            )])
                            fig.update_layout(
                                title=dict(text=f"Distribution de {cat_col}", x=0.5, xanchor='center', font=dict(color='white')),
                                height=400,
                                showlegend=True,
                                legend=dict(font=dict(color='white')),
                                paper_bgcolor='rgba(10, 14, 39, 0.5)',
                                plot_bgcolor='rgba(10, 14, 39, 0.5)',
                                font=dict(color='white')
                            )
                            st.plotly_chart(fig, use_container_width=True, key="donut")
                        else:
                            st.warning(f"No data to display for {cat_col}")
                    elif len(numeric_cols) >= 1:
                        # Fallback: bin numeric column
                        col_to_bin = numeric_cols[0]
                        df_temp = viz_data[[col_to_bin]].copy()
                        df_temp['bins'] = pd.cut(df_temp[col_to_bin], bins=5)
                        vc = df_temp['bins'].value_counts().reset_index()
                        vc.columns = ['Range', 'count']
                        vc['Range'] = vc['Range'].astype(str)
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=vc['Range'],
                            values=vc['count'],
                            hole=0.5,
                            marker=dict(
                                colors=px.colors.qualitative.Set3,
                                line=dict(color='rgba(10, 14, 39, 0.8)', width=2)
                            ),
                            textfont=dict(color='white')
                        )])
                        fig.update_layout(
                            title=f"Distribution de {col_to_bin}",
                            height=400,
                            paper_bgcolor='rgba(10, 14, 39, 0.5)',
                            plot_bgcolor='rgba(10, 14, 39, 0.5)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="donut_numeric")
                    else:
                        st.info("No suitable columns for Donut chart")
                
                with col2:
                    st.markdown("### 📊 Barre 3D")
                    if len(numeric_cols) >= 2:
                        fig = go.Figure()
                        colors = ['rgba(13, 110, 253, 0.8)', 'rgba(32, 201, 151, 0.8)', 'rgba(255, 193, 7, 0.8)']
                        for i, col in enumerate(numeric_cols[:3]):
                            fig.add_trace(go.Bar(
                                name=col,
                                x=['Min', 'Moy', 'Max'],
                                y=[viz_data[col].min(), viz_data[col].mean(), viz_data[col].max()],
                                text=[f"{viz_data[col].min():.1f}", f"{viz_data[col].mean():.1f}", f"{viz_data[col].max():.1f}"],
                                textposition='outside',
                                marker=dict(
                                    color=colors[i % 3],
                                    line=dict(color='white', width=2)
                                ),
                                textfont=dict(color='white')
                            ))
                        fig.update_layout(
                            barmode='group',
                            height=400,
                            title="Statistiques 3D",
                            paper_bgcolor='rgba(10, 14, 39, 0.5)',
                            plot_bgcolor='rgba(10, 14, 39, 0.5)',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="3dbar")
                
                st.markdown("---")
                
                # Row 2: Line + Area (lines 658-716)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Courbe")
                    if numeric_cols:
                        df_plot = df_selected.copy()
                        df_plot['Index'] = range(len(df_plot))
                        fig = go.Figure()
                        colors = ['#0d6efd', '#20c997', '#ffc107', '#0dcaf0']
                        for i, col in enumerate(numeric_cols[:4]):
                            fig.add_trace(go.Scatter(
                                x=df_plot['Index'],
                                y=df_plot[col],
                                mode='lines+markers',
                                name=col,
                                line=dict(color=colors[i % 4], width=3),
                                marker=dict(size=6)
                            ))
                        fig.update_layout(
                            title="Tendances",
                            height=400,
                            hovermode='x unified',
                            paper_bgcolor='rgba(10, 14, 39, 0.5)',
                            plot_bgcolor='rgba(10, 14, 39, 0.5)',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="line")
                
                with col2:
                    st.markdown("### 📊 Zone")
                    if numeric_cols:
                        df_plot = df_selected.copy()
                        df_plot['Index'] = range(len(df_plot))
                        fig = go.Figure()
                        colors = ['rgba(13, 110, 253, 0.6)', 'rgba(32, 201, 151, 0.6)', 'rgba(255, 193, 7, 0.6)']
                        for i, col in enumerate(numeric_cols[:3]):
                            fig.add_trace(go.Scatter(
                                x=df_plot['Index'],
                                y=df_plot[col],
                                mode='lines',
                                name=col,
                                fill='tonexty' if i > 0 else 'tozeroy',
                                line=dict(width=0.5),
                                fillcolor=colors[i % 3]
                            ))
                        fig.update_layout(
                            title="Zone Empilé",
                            height=400,
                            hovermode='x unified',
                            paper_bgcolor='rgba(10, 14, 39, 0.5)',
                            plot_bgcolor='rgba(10, 14, 39, 0.5)',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="area")
                
                st.markdown("---")
                
                # Row 3: Histogram + Gauge (lines 720-772)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Histogramme")
                    if numeric_cols:
                        col_name = numeric_cols[0]
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=df_selected[col_name],
                            nbinsx=30,
                            marker=dict(
                                color='rgba(13, 110, 253, 0.7)',
                                line=dict(color='white', width=1)
                            ),
                            name=col_name
                        ))
                        
                        mean, std = df_selected[col_name].mean(), df_selected[col_name].std()
                        x_range = np.linspace(df_selected[col_name].min(), df_selected[col_name].max(), 100)
                        y_norm = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)) * len(df_selected) * (df_selected[col_name].max() - df_selected[col_name].min()) / 30
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_norm,
                            mode='lines',
                            name='Normal',
                            line=dict(color='#ffc107', width=3, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"Distribution {col_name}",
                            height=400,
                            showlegend=True,
                            paper_bgcolor='rgba(10, 14, 39, 0.5)',
                            plot_bgcolor='rgba(10, 14, 39, 0.5)',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="hist")
                
                with col2:
                    st.markdown("### 🎯 Jauge")
                    if numeric_cols:
                        col_name = numeric_cols[0]
                        val = df_selected[col_name].mean()
                        max_val = df_selected[col_name].max()
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=val,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"Moyenne {col_name}", 'font': {'color': 'white'}},
                            delta={'reference': df_selected[col_name].median()},
                            gauge={
                                'axis': {'range': [None, max_val], 'tickcolor': 'white'},
                                'bar': {'color': "#0d6efd"},
                                'steps': [
                                    {'range': [0, max_val * 0.33], 'color': "rgba(255, 255, 255, 0.1)"},
                                    {'range': [max_val * 0.33, max_val * 0.66], 'color': "rgba(255, 255, 255, 0.2)"}
                                ],
                                'threshold': {
                                    'line': {'color': "#ff6b6b", 'width': 4},
                                    'thickness': 0.75,
                                    'value': max_val * 0.9
                                }
                            },
                            number={'font': {'color': 'white'}}
                        ))
                        fig.update_layout(
                            height=400,
                            paper_bgcolor='rgba(10, 14, 39, 0.5)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig, use_container_width=True, key="gauge")
                
                st.markdown("---")
                
                # Data Table (lines 776-785)
                st.subheader("📋 Data Table")
                st.info(f"📊 {len(all_selected_cols)} columns")
                st.dataframe(df_selected.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'),
                           use_container_width=True, height=400)
                
                # Download buttons (lines 787-809)
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df_selected.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 CSV", csv, "data.csv", "text/csv", use_container_width=True)
                with col2:
                    try:
                        import io
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                            df_selected.to_excel(writer, index=False)
                        buf.seek(0)
                        st.download_button("📊 Excel", buf, "data.xlsx", "application/vnd.ms-excel", use_container_width=True)
                    except:
                        pass
                with col3:
                    json_str = df_selected.to_json(orient='records', indent=2)
                    st.download_button("📋 JSON", json_str, "data.json", "application/json", use_container_width=True)

# Footer (matching beautiful_dashboard.py lines 811-816)
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p><strong>🤖 Powered by AI-Driven Analytics</strong></p>
        <p>Created with ❤️ using Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)
