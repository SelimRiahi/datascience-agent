"""
📊 BEAUTIFUL MULTI-PAGE ANALYTICS DASHBOARD
Professional design with complete PowerBI elements
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Page config
st.set_page_config(
    page_title="📊 Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Beautiful styling
st.markdown("""
    <style>
    /* Main background - Beautiful gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="powerbi_gauge")a - Semi-transparent white */
    .block-container {
        background-color: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Headers - Dark and bold */
    h1 {
        color: #1a1a1a !important;
        font-weight: 800;
        text-shadow: none;
    }
    
    h2 {
        color: #2c3e50 !important;
        font-weight: 700;
        margin-top: 2rem;
    }
    
    h3 {
        color: #34495e !important;
        font-weight: 600;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Make sure all text is dark */
    p {
        color: #2c3e50 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        background-color: rgba(255,255,255,0.1);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.3s;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background-color: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    
    .kpi-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(17, 153, 142, 0.3);
        transition: transform 0.3s;
    }
    
    .kpi-card-green:hover {
        transform: translateY(-5px);
    }
    
    .kpi-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(240, 147, 251, 0.3);
        transition: transform 0.3s;
    }
    
    .kpi-card-orange:hover {
        transform: translateY(-5px);
    }
    
    .kpi-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(79, 172, 254, 0.3);
        transition: transform 0.3s;
    }
    
    .kpi-card-blue:hover {
        transform: translateY(-5px);
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
    
    /* AI Recommendation Box - Beautiful and readable */
    .recommendation-box {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #d35400;
        margin: 1.5rem 0;
        color: white;
        box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);
    }
    
    .recommendation-box h4 {
        color: white !important;
        margin-top: 0;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    .recommendation-box p {
        color: white !important;
        margin-bottom: 0;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stMetricValue"] {
        color: white;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: white;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
try:
    df = pd.read_csv("temp_data.csv")
    with open("temp_recommendations.json", "r") as f:
        rec_data = json.load(f)
    
    # Handle both formats
    if 'column_groups' in rec_data:
        column_groups = rec_data['column_groups']
        all_recommendations = rec_data['all_recommendations']
    else:
        column_groups = [list(df.columns)]
        all_recommendations = [{'group_id': 1, 'columns': list(df.columns), 'recommendations': rec_data}]
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.title("🎯 NAVIGATION")
    st.markdown("---")
    page = st.radio("", ["📊 Statistics Report", "🎛️ PowerBI Dashboard"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📈 Dashboard Info")
    st.info(f"**Rows:** {len(df):,}")
    st.info(f"**Columns:** {len(df.columns)}")
    st.info(f"**Groups:** {len(column_groups)}")

# ==================== PAGE 1: STATISTICS REPORT ====================
if page == "📊 Statistics Report":
    st.title("📊 Advanced Statistics Report")
    st.markdown("**AI-Powered Data Analysis with Beautiful Visualizations**")
    st.markdown("---")
    
    # Show total number of groups
    if len(column_groups) > 1:
        st.info(f"📊 **Analyzing {len(column_groups)} Column Groups** - Each group is analyzed separately for better insights!")
    
    # Process each group
    for group_data in all_recommendations:
        group_id = group_data['group_id']
        group_cols = group_data['columns']
        recommendations = group_data['recommendations']
        
        # Create an expandable section for each group (collapsed by default for groups > 1)
        with st.expander(f"📊 **GROUP {group_id}**: {', '.join(group_cols)}", expanded=(len(column_groups) == 1)):
            # AI Recommendation with beautiful styling
            if recommendations and 'overall_recommendation' in recommendations:
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>🤖 AI Recommendation</h4>
                    <p>{recommendations['overall_recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary Statistics Table
            st.subheader("📈 Summary Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Descriptive Statistics**")
                st.dataframe(df[group_cols].describe().round(2), use_container_width=True)
            
            with col2:
                st.markdown("**📉 Additional Metrics**")
                metrics_df = pd.DataFrame({
                    'Metric': ['Count', 'Missing', 'Unique', 'Mode'],
                    **{col: [
                        df[col].count(),
                        df[col].isnull().sum(),
                        df[col].nunique(),
                        df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
                    ] for col in group_cols}
                })
                st.dataframe(metrics_df, use_container_width=True)
            
            st.markdown("---")
            
            # Correlation Matrix (if multiple numeric columns)
            numeric_cols = [col for col in group_cols if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) >= 2:
                st.subheader("🔗 Correlation Matrix")
                corr_matrix = df[numeric_cols].corr()
                
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
                fig.update_layout(
                    title="Correlation Heatmap",
                    height=400,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f"corr_heatmap_group_{group_id}")
                st.markdown("---")
            
            # Individual Column Visualizations - Enhanced
            st.subheader("📊 Individual Column Analysis")
            
            if recommendations and 'individual_visualizations' in recommendations:
                for viz in recommendations['individual_visualizations']:
                    col_name = viz['column']
                    viz_type = viz['viz_type']
                    
                    st.markdown(f"### 📌 {col_name}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create 2x2 grid of visualizations for each column
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Distribution', 'Box Plot', 'Violin Plot', 'Density Curve'),
                            specs=[[{"type": "histogram"}, {"type": "box"}],
                                   [{"type": "violin"}, {"type": "scatter"}]]
                        )
                        
                        # Histogram
                        fig.add_trace(
                            go.Histogram(x=df[col_name], name='Distribution', marker_color='#667eea'),
                            row=1, col=1
                        )
                        
                        # Box Plot
                        fig.add_trace(
                            go.Box(y=df[col_name], name='Box Plot', marker_color='#11998e'),
                            row=1, col=2
                        )
                        
                        # Violin Plot
                        fig.add_trace(
                            go.Violin(y=df[col_name], name='Violin', fillcolor='#f093fb', line_color='#f5576c'),
                            row=2, col=1
                        )
                        
                        # Density curve
                        hist, bins = np.histogram(df[col_name].dropna(), bins=30, density=True)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        fig.add_trace(
                            go.Scatter(x=bin_centers, y=hist, mode='lines', fill='tozeroy',
                                       name='Density', line=dict(color='#4facfe', width=3)),
                            row=2, col=2
                        )
                        
                        fig.update_layout(
                            height=600,
                            showlegend=False,
                            title_text=f"Complete Analysis: {col_name}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"individual_{group_id}_{col_name}_grid")
                    
                    with col2:
                        st.markdown("**🤖 AI Insights**")
                        if 'reasoning' in viz:
                            st.info(viz['reasoning'])
                        
                        st.markdown("**📊 Key Statistics**")
                        st.metric("Mean", f"{df[col_name].mean():.2f}")
                        st.metric("Median", f"{df[col_name].median():.2f}")
                        st.metric("Std Dev", f"{df[col_name].std():.2f}")
                        st.metric("Range", f"{df[col_name].max() - df[col_name].min():.2f}")
                    
                    st.markdown("---")
            
            # Relationship Visualizations - Enhanced
            if recommendations and 'relationship_visualizations' in recommendations:
                st.subheader("🔗 Advanced Relationship Analysis")
                
                for viz in recommendations['relationship_visualizations']:
                    cols = viz['columns']
                    
                    st.markdown(f"### 🔗 {cols[0]} vs {cols[1]}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create multi-view relationship analysis
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Scatter Plot', '2D Density', 'Hexbin Distribution', 'Joint Distribution'),
                            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                                   [{"type": "histogram2d"}, {"type": "scatter"}]]
                        )
                        
                        # Scatter plot
                        fig.add_trace(
                            go.Scatter(x=df[cols[0]], y=df[cols[1]], mode='markers',
                                       marker=dict(color='#667eea', size=8, opacity=0.6),
                                       name='Data Points'),
                            row=1, col=1
                        )
                        
                        # Scatter with trend
                        z = np.polyfit(df[cols[0]], df[cols[1]], 1)
                        p = np.poly1d(z)
                        fig.add_trace(
                            go.Scatter(x=df[cols[0]], y=df[cols[1]], mode='markers',
                                       marker=dict(color='#11998e', size=8, opacity=0.6),
                                       name='With Trend'),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(x=sorted(df[cols[0]]), y=p(sorted(df[cols[0]])),
                                       mode='lines', line=dict(color='red', width=3),
                                       name='Trend'),
                            row=1, col=2
                        )
                        
                        # 2D Histogram
                        fig.add_trace(
                            go.Histogram2d(x=df[cols[0]], y=df[cols[1]], colorscale='Blues'),
                            row=2, col=1
                        )
                        
                        # Joint plot style
                        fig.add_trace(
                            go.Scatter(x=df[cols[0]], y=df[cols[1]], mode='markers',
                                       marker=dict(color=df[cols[1]], colorscale='Viridis',
                                                   size=10, showscale=True),
                                       name='Joint'),
                            row=2, col=2
                        )
                        
                        fig.update_layout(
                            height=700,
                            showlegend=False,
                            title_text=f"Relationship Analysis: {cols[0]} vs {cols[1]}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"relationship_{group_id}_{cols[0]}_{cols[1]}_grid")
                    
                    with col2:
                        st.markdown("**🤖 AI Insights**")
                        if 'reasoning' in viz:
                            st.info(viz['reasoning'])
                        
                        st.markdown("**📈 Correlation Analysis**")
                        corr = df[cols[0]].corr(df[cols[1]])
                        st.metric("Correlation", f"{corr:.4f}")
                        
                        if abs(corr) > 0.7:
                            st.success("🟢 Strong correlation")
                        elif abs(corr) > 0.4:
                            st.warning("🟡 Moderate correlation")
                        else:
                            st.error("🔴 Weak correlation")
                        
                        st.markdown("**🔍 Insights to Explore**")
                        if 'insights_to_look_for' in viz:
                            for insight in viz['insights_to_look_for']:
                                st.markdown(f"• {insight}")
                    
                    st.markdown("---")

# ==================== PAGE 2: POWERBI DASHBOARD ====================
else:
    st.title("🎛️ PowerBI-Style Interactive Dashboard")
    st.markdown("**Professional Business Intelligence Visualizations**")
    st.markdown("---")
    
    # Get all unique columns from all groups (no duplicates)
    all_selected_cols = []
    seen_cols = set()
    for group in column_groups:
        for col in group:
            if col not in seen_cols:
                all_selected_cols.append(col)
                seen_cols.add(col)
    
    # Filter to only selected columns
    df_selected = df[all_selected_cols]
    
    # Show info about groups
    if len(column_groups) > 1:
        st.info(f"📊 **Displaying {len(all_selected_cols)} unique columns** from {len(column_groups)} groups (duplicates removed)")
    
    # KPI Cards Section
    numeric_cols = [col for col in all_selected_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    if numeric_cols:
        st.subheader("📈 Key Performance Indicators")
        
        cols_display = st.columns(min(4, len(numeric_cols)))
        colors = ['kpi-card', 'kpi-card-green', 'kpi-card-orange', 'kpi-card-blue']
        
        for i, col in enumerate(numeric_cols[:4]):
            with cols_display[i]:
                avg_val = df[col].mean()
                display_val = f"{avg_val:,.2f}" if avg_val < 1000 else f"{avg_val:,.0f}"
                
                st.markdown(f"""
                <div class="{colors[i % 4]}">
                    <div class="kpi-value">{display_val}</div>
                    <div class="kpi-label">Average {col}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # PowerBI Visualizations Grid
    st.subheader("📊 PowerBI Visualizations")
    
    # Row 1: Donut (Anneau) + 3D Bar (Barre 3D)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🍩 Anneau (Donut Chart)")
        cat_cols = [col for col in all_selected_cols if df[col].dtype == 'object' or df[col].nunique() < 20]
        
        if cat_cols:
            cat_col = cat_cols[0]
            value_counts = df[cat_col].value_counts().reset_index()
            value_counts.columns = [cat_col, 'count']
            
            fig = go.Figure(data=[go.Pie(
                labels=value_counts[cat_col],
                values=value_counts['count'],
                hole=0.5,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textposition='outside',
                pull=[0.1 if i == 0 else 0 for i in range(len(value_counts))]
            )])
            fig.update_layout(
                title=dict(text=f"Distribution de {cat_col}", x=0.5, xanchor='center'),
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
            )
            st.plotly_chart(fig, use_container_width=True, key="powerbi_donut_categorical")
        elif len(numeric_cols) >= 1:
            # Create bins for numeric column
            col_to_bin = numeric_cols[0]
            df_temp = df.copy()
            df_temp['bins'] = pd.cut(df_temp[col_to_bin], bins=5).astype(str)
            value_counts = df_temp['bins'].value_counts().reset_index()
            value_counts.columns = ['Range', 'count']
            
            fig = go.Figure(data=[go.Pie(
                labels=value_counts['Range'],
                values=value_counts['count'],
                hole=0.5,
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig.update_layout(title=f"Distribution de {col_to_bin}", height=400)
            st.plotly_chart(fig, use_container_width=True, key="powerbi_donut_numeric")
    
    with col2:
        st.markdown("### 📊 Barre 3D (3D Bar Chart)")
        if len(numeric_cols) >= 2:
            fig = go.Figure()
            
            for i, col in enumerate(numeric_cols[:3]):
                fig.add_trace(go.Bar(
                    name=col,
                    x=['Minimum', 'Moyenne', 'Maximum'],
                    y=[df[col].min(), df[col].mean(), df[col].max()],
                    text=[f"{df[col].min():.1f}", f"{df[col].mean():.1f}", f"{df[col].max():.1f}"],
                    textposition='outside',
                    marker=dict(
                        color=[f'rgba({102 + i*50}, {126 + i*30}, {234 - i*40}, 0.8)' for _ in range(3)],
                        line=dict(color='white', width=2)
                    )
                ))
            
            fig.update_layout(
                barmode='group',
                height=400,
                title="Statistiques Comparatives 3D",
                xaxis_title="Statistique",
                yaxis_title="Valeur",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True, key="powerbi_3d_bar")
    
    st.markdown("---")
    
    # Row 2: Line Chart (Courbe) + Area Chart (Zone)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Courbe (Line Chart)")
        if len(numeric_cols) >= 1:
            df_plot = df_selected.copy()
            df_plot['Index'] = range(len(df_plot))
            
            fig = go.Figure()
            colors = ['#667eea', '#11998e', '#f093fb', '#4facfe']
            
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
                title="Tendances et Évolutions",
                height=400,
                xaxis_title="Index",
                yaxis_title="Valeur",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True, key="powerbi_line_chart")
    
    with col2:
        st.markdown("### 📊 Zone (Area Chart)")
        if len(numeric_cols) >= 1:
            df_plot = df_selected.copy()
            df_plot['Index'] = range(len(df_plot))
            
            fig = go.Figure()
            colors = ['rgba(102, 126, 234, 0.6)', 'rgba(17, 153, 142, 0.6)', 
                      'rgba(240, 147, 251, 0.6)', 'rgba(79, 172, 254, 0.6)']
            
            for i, col in enumerate(numeric_cols[:3]):
                fig.add_trace(go.Scatter(
                    x=df_plot['Index'],
                    y=df_plot[col],
                    mode='lines',
                    name=col,
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(width=0.5),
                    fillcolor=colors[i % 4]
                ))
            
            fig.update_layout(
                title="Graphique en Zone Empilé",
                height=400,
                xaxis_title="Index",
                yaxis_title="Valeur",
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True, key="powerbi_area_chart")
    
    st.markdown("---")
    
    # Row 3: Histogram (Histogramme) + Gauge (Jauge)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Histogramme")
        if numeric_cols:
            selected_col = numeric_cols[0]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_selected[selected_col],
                nbinsx=30,
                marker=dict(
                    color='#667eea',
                    line=dict(color='white', width=1)
                ),
                name=selected_col
            ))
            
            # Add normal distribution curve
            mean = df_selected[selected_col].mean()
            std = df_selected[selected_col].std()
            x_range = np.linspace(df_selected[selected_col].min(), df_selected[selected_col].max(), 100)
            y_norm = ((1 / (np.sqrt(2 * np.pi) * std)) * 
                     np.exp(-0.5 * ((x_range - mean) / std) ** 2)) * len(df_selected) * (df_selected[selected_col].max() - df_selected[selected_col].min()) / 30
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_norm,
                mode='lines',
                name='Normal Dist.',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Distribution de {selected_col}",
                height=400,
                xaxis_title=selected_col,
                yaxis_title="Fréquence",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True, key="powerbi_histogram")
    
    with col2:
        st.markdown("### 🎯 Jauge (Gauge Chart)")
        if numeric_cols:
            col_for_gauge = numeric_cols[0]
            current_value = df_selected[col_for_gauge].mean()
            max_value = df_selected[col_for_gauge].max()
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Moyenne {col_for_gauge}"},
                delta={'reference': df_selected[col_for_gauge].median()},
                gauge={
                    'axis': {'range': [None, max_value]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, max_value * 0.33], 'color': "lightgray"},
                        {'range': [max_value * 0.33, max_value * 0.66], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_value * 0.9
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="powerbi_gauge")
    
    st.markdown("---")
    
    # Data Table Section
    st.subheader("📋 Tableau de Données Détaillé")
    
    # Show only selected columns
    st.info(f"� Displaying {len(all_selected_cols)} selected columns")
    
    st.dataframe(
        df_selected.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'),
        use_container_width=True,
        height=400
    )
    
    # Download Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df_selected.to_csv(index=False).encode('utf-8')
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="dashboard_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_buffer = pd.ExcelWriter('temp.xlsx', engine='xlsxwriter')
        df_selected.to_excel(excel_buffer, index=False)
        excel_buffer.close()
        
        st.download_button(
            label="📊 Download Excel",
            data=open('temp.xlsx', 'rb').read(),
            file_name="dashboard_data.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True
        )
    
    with col3:
        json_str = df_selected.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Download JSON",
            data=json_str,
            file_name="dashboard_data.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p><strong>🤖 Powered by AI-Driven Analytics</strong></p>
        <p>Created with ❤️ using Streamlit & Plotly</p>
    </div>
    """, unsafe_allow_html=True)
