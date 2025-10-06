"""
🧹 DATA CLEANING MODULE
AI-powered intelligent data cleaning with real-time feedback
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import time
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Import the backend cleaning agent
from data_quality.data_cleaning_agent import CompleteDataCleaningAgent

# Page configuration
st.set_page_config(
    page_title="Data Cleaning - AI Agent",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic data science design
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
    }
    
    /* Animated background particles */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20% 30%, rgba(255,255,255,0.3), transparent),
            radial-gradient(2px 2px at 60% 70%, rgba(255,255,255,0.3), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(255,255,255,0.3), transparent),
            radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,0.3), transparent);
        background-size: 200% 200%;
        animation: particleFloat 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleFloat {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
    }
    
    /* Hero section - Futuristic data visualization style */
    .hero-section {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 50%, #d63384 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        border: 2px solid rgba(13, 110, 253, 0.3);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 
            0 0 60px rgba(13, 110, 253, 0.4),
            inset 0 0 60px rgba(13, 110, 253, 0.1);
        position: relative;
        overflow: hidden;
        animation: heroGlow 3s ease-in-out infinite alternate;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: heroSweep 3s ease-in-out infinite;
    }
    
    @keyframes heroGlow {
        from { box-shadow: 0 0 40px rgba(13, 110, 253, 0.3), inset 0 0 40px rgba(13, 110, 253, 0.1); }
        to { box-shadow: 0 0 80px rgba(13, 110, 253, 0.6), inset 0 0 80px rgba(13, 110, 253, 0.2); }
    }
    
    @keyframes heroSweep {
        0% { transform: rotate(0deg) translate(-50%, -50%); }
        100% { transform: rotate(360deg) translate(-50%, -50%); }
    }
    
    .hero-section h1 {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 
            0 0 10px rgba(255,255,255,0.5),
            0 0 20px rgba(13, 110, 253, 0.8),
            0 0 30px rgba(13, 110, 253, 0.6);
        letter-spacing: 3px;
        position: relative;
        z-index: 1;
    }
    
    .hero-section p {
        font-size: 1.3rem;
        margin-top: 1rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Cyber card styling - Neural network inspired */
    .glass-card {
        background: rgba(10, 14, 39, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(13, 110, 253, 0.3);
        box-shadow: 
            0 8px 32px 0 rgba(13, 110, 253, 0.2),
            inset 0 0 20px rgba(13, 110, 253, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(13, 110, 253, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .glass-card:hover::before {
        left: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(13, 110, 253, 0.6);
        box-shadow: 
            0 15px 50px 0 rgba(13, 110, 253, 0.4),
            inset 0 0 30px rgba(13, 110, 253, 0.1);
    }
    
    /* Metric cards - Data dashboard style */
    .metric-card {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        border: 2px solid rgba(13, 110, 253, 0.4);
        box-shadow: 
            0 8px 25px rgba(13, 110, 253, 0.3),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: metricPulse 4s ease-in-out infinite;
    }
    
    @keyframes metricPulse {
        0%, 100% { transform: scale(0.8); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 0.8; }
    }
    
    .metric-card:hover {
        transform: scale(1.08) rotate(-2deg);
        box-shadow: 
            0 12px 40px rgba(13, 110, 253, 0.5),
            inset 0 0 30px rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0.5rem 0;
        text-shadow: 0 0 10px rgba(255,255,255,0.5);
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        position: relative;
        z-index: 1;
    }
    
    /* Progress step indicator - Data pipeline visualization */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }
    
    .step {
        flex: 1;
        text-align: center;
        position: relative;
    }
    
    .step-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.4rem;
        font-family: 'Orbitron', sans-serif;
        border: 3px solid rgba(13, 110, 253, 0.5);
        box-shadow: 
            0 0 20px rgba(13, 110, 253, 0.5),
            inset 0 0 10px rgba(255, 255, 255, 0.2);
    }
    
    .step-circle.inactive {
        background: rgba(30, 40, 70, 0.5);
        border-color: rgba(100, 100, 100, 0.3);
        box-shadow: none;
        color: #666;
    }
    
    .step-circle.active {
        animation: stepPulse 1.5s ease-in-out infinite;
        box-shadow: 
            0 0 30px rgba(13, 110, 253, 0.8),
            0 0 60px rgba(13, 110, 253, 0.4),
            inset 0 0 20px rgba(255, 255, 255, 0.3);
    }
    
    @keyframes stepPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 30px rgba(13, 110, 253, 0.8), inset 0 0 20px rgba(255, 255, 255, 0.3);
        }
        50% { 
            transform: scale(1.15);
            box-shadow: 0 0 50px rgba(13, 110, 253, 1), inset 0 0 30px rgba(255, 255, 255, 0.5);
        }
    }
    
    .step-label {
        display: block;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        font-weight: 600;
        color: #0d6efd;
        text-shadow: 0 0 10px rgba(13, 110, 253, 0.5);
    }
    
    /* Animations - Data flow inspired */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(-30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(-40px); 
        }
        to { 
            opacity: 1; 
            transform: translateX(0); 
        }
    }
    
    @keyframes dataFlow {
        0% { 
            background-position: 0% 50%; 
        }
        50% { 
            background-position: 100% 50%; 
        }
        100% { 
            background-position: 0% 50%; 
        }
    }
    
    /* Buttons - Cyberpunk style */
    .stButton > button {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        color: white;
        border: 2px solid rgba(13, 110, 253, 0.5);
        border-radius: 12px;
        padding: 0.9rem 2.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 
            0 4px 20px rgba(13, 110, 253, 0.4),
            inset 0 0 15px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 
            0 8px 30px rgba(13, 110, 253, 0.6),
            inset 0 0 25px rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02);
    }
    
    /* File uploader - Neural interface style */
    .uploadedFile {
        background: rgba(10, 14, 39, 0.6);
        border: 2px dashed rgba(13, 110, 253, 0.5);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: rgba(13, 110, 253, 0.8);
        background: rgba(10, 14, 39, 0.8);
        box-shadow: 0 0 30px rgba(13, 110, 253, 0.3);
    }
    
    /* Status badges - Data label style */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.85rem;
        margin: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 2px solid;
        box-shadow: 0 0 15px currentColor;
        animation: badgeGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes badgeGlow {
        from { box-shadow: 0 0 10px currentColor; }
        to { box-shadow: 0 0 20px currentColor; }
    }
    
    .status-success {
        background: rgba(25, 135, 84, 0.2);
        color: #20c997;
        border-color: #20c997;
    }
    
    .status-warning {
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border-color: #ffc107;
    }
    
    .status-error {
        background: rgba(220, 53, 69, 0.2);
        color: #ff6b6b;
        border-color: #ff6b6b;
    }
    
    .status-info {
        background: rgba(13, 110, 253, 0.2);
        color: #0dcaf0;
        border-color: #0dcaf0;
    }
    
    /* Data preview table - Matrix style */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid rgba(13, 110, 253, 0.3);
        box-shadow: 0 0 25px rgba(13, 110, 253, 0.2);
        background: rgba(10, 14, 39, 0.6);
    }
    
    /* Tabs styling - Data pipeline sections */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(10, 14, 39, 0.4);
        border-radius: 15px;
        padding: 0.5rem;
        border: 1px solid rgba(13, 110, 253, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 2px solid transparent;
        border-radius: 10px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(13, 110, 253, 0.1);
        color: #0dcaf0;
        border-color: rgba(13, 110, 253, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        color: white;
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 0 20px rgba(13, 110, 253, 0.5);
    }
    
    /* Expander styling - Collapsible data sections */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        color: white;
        border-radius: 12px;
        font-weight: 700;
        border: 2px solid rgba(13, 110, 253, 0.5);
        box-shadow: 0 0 20px rgba(13, 110, 253, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        box-shadow: 0 0 30px rgba(13, 110, 253, 0.5);
        transform: translateX(5px);
    }
    
    /* Info boxes - Neural alert style */
    .info-box {
        background: linear-gradient(135deg, rgba(13, 110, 253, 0.2) 0%, rgba(13, 202, 240, 0.2) 100%);
        color: #0dcaf0;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(13, 202, 240, 0.4);
        box-shadow: 0 0 25px rgba(13, 202, 240, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 152, 0, 0.2) 100%);
        color: #ffc107;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(255, 193, 7, 0.4);
        box-shadow: 0 0 25px rgba(255, 193, 7, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(25, 135, 84, 0.2) 0%, rgba(32, 201, 151, 0.2) 100%);
        color: #20c997;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(32, 201, 151, 0.4);
        box-shadow: 0 0 25px rgba(32, 201, 151, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling - Control panel */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(13, 110, 253, 0.3);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #e0e0e0;
    }
    
    /* Text colors for dark theme */
    .stMarkdown, p, span, label {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(13, 110, 253, 0.5);
    }
    
    /* Radio buttons - Modern selector */
    .stRadio > div {
        background: rgba(10, 14, 39, 0.6);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(13, 110, 253, 0.3);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        color: #0dcaf0;
        text-shadow: 0 0 10px rgba(13, 202, 240, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cleaning_agent' not in st.session_state:
    st.session_state.cleaning_agent = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'semantic_types' not in st.session_state:
    st.session_state.semantic_types = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'execution_log' not in st.session_state:
    st.session_state.execution_log = []
if 'current_df' not in st.session_state:
    st.session_state.current_df = None

# Sync data from main app.py if available
if 'raw_data' in st.session_state and st.session_state.raw_data is not None:
    if st.session_state.original_df is None:
        st.session_state.original_df = st.session_state.raw_data.copy()
        st.session_state.current_df = st.session_state.raw_data.copy()

# Hero Section
st.markdown("""
    <div class="hero-section">
        <h1>⚡ NEURAL DATA CLEANER</h1>
        <p>AI-Powered Quantum Data Purification Engine</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIGURATION")
    
    sample_size = st.slider(
        "Neural Sample Size",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of sample rows for AI semantic analysis"
    )
    
    st.markdown("---")
    st.markdown("### 📊 LIVE METRICS")
    
    if st.session_state.original_df is not None:
        st.metric("📥 Original Rows", f"{len(st.session_state.original_df):,}")
        st.metric("📋 Columns", len(st.session_state.original_df.columns))
        
        if st.session_state.cleaned_df is not None:
            st.metric("✨ Cleaned Rows", f"{len(st.session_state.cleaned_df):,}")
            improvement = len(st.session_state.original_df) - len(st.session_state.cleaned_df)
            st.metric("⚡ Processed", f"{improvement:,}")
    
    st.markdown("---")
    st.markdown("### 🎯 SYSTEM STATUS")
    st.markdown("""
    <div class="info-box" style="font-size: 0.85rem; padding: 1rem;">
        <strong>🧠 Neural Cleaning System</strong><br><br>
        ⚡ Semantic data understanding<br>
        🎯 Smart action recommendations<br>
        🔒 Safe execution protocols<br>
        📊 Real-time analytics
    </div>
    """, unsafe_allow_html=True)

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Upload & Initialize",
    "🔍 Semantic Analysis", 
    "🛠️ Cleaning Actions",
    "📊 Results & Export"
])

# TAB 1: Upload & Initialize
with tab1:
    st.markdown("## 📂 Upload Your Dataset")
    
    # Check if data was uploaded from main page
    if st.session_state.original_df is not None:
        st.success(f"✅ Data loaded from main page: **{len(st.session_state.original_df)} rows × {len(st.session_state.original_df.columns)} columns**")
        st.info("👉 Initialize the AI Agent below or upload a different file.")
        
        # Show data preview for loaded data
        st.markdown("### 👀 Data Preview")
        st.dataframe(st.session_state.original_df.head(10), use_container_width=True, height=300)
        
        # Display metrics for loaded data
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Rows</div>
                    <div class="metric-value">{len(st.session_state.original_df):,}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Columns</div>
                    <div class="metric-value">{len(st.session_state.original_df.columns)}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            missing_pct = (st.session_state.original_df.isnull().sum().sum() / (len(st.session_state.original_df) * len(st.session_state.original_df.columns)) * 100)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Missing Data</div>
                    <div class="metric-value">{missing_pct:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file (CSV, Excel, or XLSX)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset to begin the cleaning process"
        )
        
        if uploaded_file is not None:
            try:
                # Load the dataframe
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.original_df = df
                st.session_state.current_df = df.copy()
                
                st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
                
                # Display metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Rows</div>
                            <div class="metric-value">{len(df):,}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Columns</div>
                            <div class="metric-value">{len(df.columns)}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Missing Data</div>
                            <div class="metric-value">{missing_pct:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Data preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True, height=300)
                
                # Data quality overview
                st.markdown("### 🔍 Quick Data Quality Check")
                
                col_q1, col_q2 = st.columns(2)
                
                with col_q1:
                    st.markdown("**Missing Values per Column**")
                    missing_df = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum().values,
                        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
                    })
                    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                    
                    if len(missing_df) > 0:
                        st.dataframe(missing_df, use_container_width=True, height=200)
                    else:
                        st.success("✅ No missing values detected!")
                
                with col_q2:
                    st.markdown("**Data Types Distribution**")
                    dtype_counts = df.dtypes.value_counts()
                    
                    fig = px.pie(
                        values=dtype_counts.values,
                        names=dtype_counts.index.astype(str),
                        title="Column Data Types",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Purples_r
                    )
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.markdown("### 📝 Quick Tips")
        st.markdown("""
        <div class="info-box">
            <strong>Supported Formats:</strong><br>
            📄 CSV Files<br>
            📊 Excel (.xlsx, .xls)<br><br>
            <strong>Best Practices:</strong><br>
            ✓ Clean column headers<br>
            ✓ Consistent data types<br>
            ✓ Remove extra spaces<br>
            ✓ Valid date formats
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize button - shown whenever there's data (from upload or main page)
    if st.session_state.original_df is not None:
        st.markdown("---")
        
        if st.session_state.cleaning_agent is not None:
            st.success("✅ AI Agent is already initialized!")
            st.info("👉 Move to the **Semantic Analysis** tab to continue")
            if st.button("🔄 Reinitialize Agent", use_container_width=True):
                with st.spinner("🤖 Reinitializing AI Agent..."):
                    try:
                        st.session_state.cleaning_agent = CompleteDataCleaningAgent()
                        st.session_state.current_step = 1
                        st.success("✅ AI Agent reinitialized successfully!")
                    except Exception as e:
                        st.error(f"❌ Error reinitializing agent: {str(e)}")
        else:
            col_init1, col_init2 = st.columns(2)
            
            with col_init1:
                if st.button("🚀 Initialize AI Cleaning Agent", type="primary", use_container_width=True):
                    with st.spinner("🤖 Initializing AI Agent..."):
                        try:
                            st.session_state.cleaning_agent = CompleteDataCleaningAgent()
                            st.session_state.current_step = 1
                            st.success("✅ AI Agent initialized successfully!")
                            st.info("👉 Move to the **Semantic Analysis** tab to continue")
                        except Exception as e:
                            st.error(f"❌ Error initializing agent: {str(e)}")
            
            with col_init2:
                if st.button("⏩ Skip Cleaning (Use Raw Data)", use_container_width=True):
                    st.session_state.cleaned_df = st.session_state.original_df.copy()
                    st.session_state.cleaning_complete = True
                    st.session_state.pipeline_stage = 'cleaning'
                    st.success("✅ Skipped cleaning - Using raw data for ML training")
                    time.sleep(0.5)
                    st.switch_page("pages/2_🧠_ML_Training.py")
        
        # Show skip option even if agent is initialized
        if st.session_state.cleaning_agent is not None and not st.session_state.cleaning_complete:
            st.markdown("---")
            st.markdown("""
            <div class="warning-box">
                <strong>⏩ Want to skip AI cleaning?</strong><br>
                You can proceed directly to ML training with the current data.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("⏩ Skip All Cleaning Steps", use_container_width=True):
                st.session_state.cleaned_df = st.session_state.current_df.copy()
                st.session_state.cleaning_complete = True
                st.session_state.pipeline_stage = 'cleaning'
                st.success("✅ Cleaning skipped - Ready for ML training")
                time.sleep(0.5)
                st.switch_page("pages/2_🧠_ML_Training.py")

# TAB 2: Semantic Analysis
with tab2:
    st.markdown("## 🔍 Semantic Understanding & Analysis")
    
    if st.session_state.original_df is None:
        st.warning("⚠️ Please upload a dataset in the **Upload & Initialize** tab first")
    elif st.session_state.cleaning_agent is None:
        st.warning("⚠️ Please initialize the AI Agent in the **Upload & Initialize** tab first")
    else:
        st.markdown("""
            <div class="info-box">
                <strong>Step 1: Semantic Understanding</strong><br>
                The AI agent will analyze your data to understand what each column represents 
                in a business context (e.g., customer age, product price, email address, etc.)
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧠 Start Semantic Analysis", type="primary", use_container_width=True):
            with st.spinner("🔍 Analyzing data semantics..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Semantic Understanding
                    status_text.text("Understanding column semantics...")
                    progress_bar.progress(30)
                    
                    semantic_types = st.session_state.cleaning_agent.get_semantic_understanding(
                        st.session_state.original_df,
                        sample_size=sample_size
                    )
                    st.session_state.semantic_types = semantic_types
                    
                    progress_bar.progress(60)
                    status_text.text("Generating cleaning recommendations...")
                    
                    # Step 2: Generate Recommendations
                    recommendations = st.session_state.cleaning_agent.generate_cleaning_recommendations(
                        st.session_state.original_df,
                        semantic_types
                    )
                    st.session_state.recommendations = recommendations
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    st.success("✅ Semantic analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        
        # Display results if available
        if st.session_state.semantic_types is not None:
            st.markdown("---")
            st.markdown("### 🎯 Semantic Analysis Results")
            
            # Create a beautiful display of semantic types
            semantic_df = pd.DataFrame([
                {
                    'Column': col,
                    'Data Type': str(st.session_state.original_df[col].dtype),
                    'Semantic Type': sem_type,
                    'Sample Values': ', '.join(str(x) for x in st.session_state.original_df[col].dropna().head(3).tolist())
                }
                for col, sem_type in st.session_state.semantic_types.items()
            ])
            
            st.dataframe(
                semantic_df,
                use_container_width=True,
                height=400,
                column_config={
                    "Column": st.column_config.TextColumn("Column Name", width="medium"),
                    "Data Type": st.column_config.TextColumn("Data Type", width="small"),
                    "Semantic Type": st.column_config.TextColumn("Semantic Type", width="medium"),
                    "Sample Values": st.column_config.TextColumn("Sample Values", width="large")
                }
            )
            
            # Show recommendations summary
            if st.session_state.recommendations is not None:
                st.markdown("### 🛠️ Cleaning Recommendations Summary")
                
                total_suggestions = sum(
                    len(rec['cleaning_logic'].get('cleaning_suggestions', []))
                    for rec in st.session_state.recommendations['cleaning_recommendations'].values()
                )
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    st.metric("Columns Analyzed", len(st.session_state.semantic_types))
                
                with col_s2:
                    st.metric("Total Suggestions", total_suggestions)
                
                with col_s3:
                    critical_count = sum(
                        1 for rec in st.session_state.recommendations['cleaning_recommendations'].values()
                        for sug in rec['cleaning_logic'].get('cleaning_suggestions', [])
                        if sug.get('priority') == 'critical'
                    )
                    st.metric("Critical Issues", critical_count)
                
                st.info("👉 Move to the **Cleaning Actions** tab to review and apply cleaning operations")

# TAB 3: Cleaning Actions  
with tab3:
    st.markdown("## 🛠️ Interactive Cleaning Actions")
    
    if st.session_state.recommendations is None:
        st.warning("⚠️ Please complete the semantic analysis in the **Semantic Analysis** tab first")
    else:
        st.markdown("""
            <div class="info-box">
                <strong>Step 2: Review & Execute Cleaning Actions</strong><br>
                Review AI-generated cleaning suggestions and choose which actions to apply to your data.
            </div>
        """, unsafe_allow_html=True)
        
        # Display cleaning recommendations by column
        for column, rec in st.session_state.recommendations['cleaning_recommendations'].items():
            semantic_type = rec['semantic_type']
            suggestions = rec['cleaning_logic'].get('cleaning_suggestions', [])
            business_rules = rec['cleaning_logic'].get('business_rules', [])
            
            if not suggestions:
                continue
            
            with st.expander(f"📋 **{column}** - {semantic_type}", expanded=False):
                # Show business rules
                if business_rules:
                    st.markdown("**🎯 Business Rules:**")
                    for rule in business_rules:
                        st.markdown(f"- {rule}")
                    st.markdown("---")
                
                # Show current column stats
                col_data = st.session_state.current_df[column]
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Total Values", len(col_data))
                with col_stats2:
                    st.metric("Missing Values", col_data.isnull().sum())
                with col_stats3:
                    st.metric("Unique Values", col_data.nunique())
                
                st.markdown("---")
                
                # Display each suggestion
                for idx, suggestion in enumerate(suggestions):
                    st.markdown(f"### 🔧 Action {idx + 1}: {suggestion.get('action', 'Unknown')}")
                    
                    col_left, col_right = st.columns([2, 1])
                    
                    with col_left:
                        st.markdown(f"**Description:** {suggestion.get('description', 'No description')}")
                        st.markdown(f"**Justification:** {suggestion.get('justification', 'No justification')}")
                        
                        # Priority badge
                        priority = suggestion.get('priority', 'medium')
                        priority_colors = {
                            'critical': 'status-error',
                            'high': 'status-warning',
                            'medium': 'status-info',
                            'low': 'status-success'
                        }
                        st.markdown(f"""
                            <span class="status-badge {priority_colors.get(priority, 'status-info')}">
                                {priority.upper()} PRIORITY
                            </span>
                        """, unsafe_allow_html=True)
                        
                        # Show alternative actions with radio selection
                        alternatives = suggestion.get('alternative_actions', [])
                        selected_action_idx = 0  # Default to primary action
                        
                        if alternatives:
                            st.markdown("**🔄 Choose Action to Execute:**")
                            action_options = [f"Primary: {suggestion.get('action', 'Unknown')}"]
                            action_options.extend([f"Alt {i+1}: {alt.get('action', 'Unknown')}" for i, alt in enumerate(alternatives[:3])])
                            
                            selected_action_idx = st.radio(
                                "Select:",
                                range(len(action_options)),
                                format_func=lambda x: action_options[x],
                                key=f"radio_{column}_{idx}",
                                horizontal=False
                            )
                            
                            # Show description of selected alternative
                            if selected_action_idx > 0:
                                st.info(f"ℹ️ {alternatives[selected_action_idx - 1].get('description', 'No description')}")
                        else:
                            st.markdown("**ℹ️ No alternative actions available**")
                    
                    with col_right:
                        # Action button
                        action_key = f"{column}_{idx}_{suggestion.get('action')}"
                        
                        if st.button(f"▶️ Execute Selected", key=action_key, use_container_width=True):
                            with st.spinner("Executing cleaning action..."):
                                try:
                                    # Determine which action to execute
                                    if selected_action_idx == 0:
                                        # Primary action
                                        chosen_action = suggestion.get("action")
                                        chosen_desc = suggestion.get("description")
                                    else:
                                        # Alternative action
                                        alt = alternatives[selected_action_idx - 1]
                                        chosen_action = alt.get("action")
                                        chosen_desc = alt.get("description")
                                    
                                    # Prepare action details
                                    action_details = {
                                        "action": chosen_action,
                                        "description": chosen_desc,
                                        "target": column,
                                        "semantic_type": semantic_type,
                                        "priority": suggestion.get("priority"),
                                        "automated": suggestion.get("automated", False)
                                    }
                                    
                                    # Generate function
                                    df_info = {
                                        "columns": st.session_state.current_df.columns.tolist(),
                                        "dtypes": {str(k): str(v) for k, v in st.session_state.current_df.dtypes.to_dict().items()},
                                        "sample_rows": min(len(st.session_state.current_df), 5)
                                    }
                                    
                                    function_code = st.session_state.cleaning_agent.generate_cleaning_function(
                                        action_details, 
                                        df_info
                                    )
                                    
                                    # Validate safety
                                    is_safe, safety_message = st.session_state.cleaning_agent.validate_function_safety(function_code)
                                    
                                    if not is_safe:
                                        st.error(f"❌ Safety check failed: {safety_message}")
                                    else:
                                        # Execute
                                        result_df, log_message = st.session_state.cleaning_agent.execute_cleaning_action(
                                            st.session_state.current_df,
                                            action_details,
                                            function_code
                                        )
                                        
                                        # Update current dataframe
                                        st.session_state.current_df = result_df
                                        st.session_state.execution_log.append({
                                            "column": column,
                                            "action": suggestion.get("action"),
                                            "log": log_message,
                                            "timestamp": datetime.now().strftime("%H:%M:%S")
                                        })
                                        
                                        st.success(f"✅ {log_message}")
                                        st.rerun()
                                        
                                except Exception as e:
                                    st.error(f"❌ Error executing action: {str(e)}")
                    
                    st.markdown("---")
        
        # Show execution log in sidebar
        if st.session_state.execution_log:
            st.sidebar.markdown("### 📝 Execution Log")
            for log_entry in reversed(st.session_state.execution_log[-5:]):
                st.sidebar.markdown(f"""
                    <div class="status-badge status-success">
                        {log_entry['timestamp']} - {log_entry['column']}: {log_entry['action']}
                    </div>
                """, unsafe_allow_html=True)

# TAB 4: Results & Export
with tab4:
    st.markdown("## 📊 Cleaning Results & Export")
    
    if st.session_state.current_df is None or st.session_state.original_df is None:
        st.warning("⚠️ No cleaning operations have been performed yet")
    else:
        # Show before/after comparison
        st.markdown("### 📈 Before & After Comparison")
        
        col_comp1, col_comp2, col_comp3, col_comp4 = st.columns(4)
        
        with col_comp1:
            st.metric(
                "Original Rows",
                f"{len(st.session_state.original_df):,}",
                delta=None
            )
        
        with col_comp2:
            current_rows = len(st.session_state.current_df)
            delta_rows = current_rows - len(st.session_state.original_df)
            st.metric(
                "Current Rows",
                f"{current_rows:,}",
                delta=f"{delta_rows:+,}"
            )
        
        with col_comp3:
            original_missing = st.session_state.original_df.isnull().sum().sum()
            current_missing = st.session_state.current_df.isnull().sum().sum()
            delta_missing = current_missing - original_missing
            st.metric(
                "Missing Values",
                f"{current_missing:,}",
                delta=f"{delta_missing:+,}",
                delta_color="inverse"
            )
        
        with col_comp4:
            actions_count = len(st.session_state.execution_log)
            st.metric(
                "Actions Applied",
                actions_count
            )
        
        # Data preview tabs
        st.markdown("### 👀 Data Preview")
        preview_tab1, preview_tab2 = st.tabs(["Original Data", "Cleaned Data"])
        
        with preview_tab1:
            st.dataframe(st.session_state.original_df.head(20), use_container_width=True, height=400)
        
        with preview_tab2:
            st.dataframe(st.session_state.current_df.head(20), use_container_width=True, height=400)
        
        # Execution summary
        if st.session_state.execution_log:
            st.markdown("### 📋 Execution Summary")
            
            summary_df = pd.DataFrame(st.session_state.execution_log)
            st.dataframe(summary_df, use_container_width=True, height=300)
        
        # Export options
        st.markdown("### 💾 Export Cleaned Data")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # Download as CSV
            csv_data = st.session_state.current_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_exp2:
            # Download as Excel
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.current_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Save to session for next steps
        if st.button("✅ Save & Continue to ML", type="primary", use_container_width=True):
            st.session_state.cleaned_df = st.session_state.current_df
            st.success("✅ Cleaned data saved! Ready for ML analysis.")
            st.info("👉 Navigate to the ML Agent page to continue")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; opacity: 0.7; padding: 2rem;">
        <p>🤖 Intelligent Data Science Agent - Data Cleaning Module</p>
        <p>Powered by AI | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
