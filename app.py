"""
🤖 INTELLIGENT DATA SCIENCE AGENT
Complete automated pipeline: Clean → ML → Visualize

Main entry point for the Streamlit multi-page application
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="AI Data Science Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/SelimRiahi/datascience-agent',
        'Report a bug': "https://github.com/SelimRiahi/datascience-agent/issues",
        'About': "# Intelligent Data Science Agent\nAutonomous end-to-end data analysis"
    }
)

# Custom CSS for better design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    /* Main theme - Dark futuristic */
    :root {
        --primary-color: #0d6efd;
        --success-color: #20c997;
        --warning-color: #ffc107;
        --error-color: #ff6b6b;
    }
    
    .main {
        background: #0a0e27;
        background-image: 
            radial-gradient(at 20% 30%, rgba(13, 110, 253, 0.15) 0px, transparent 50%),
            radial-gradient(at 80% 70%, rgba(132, 59, 206, 0.15) 0px, transparent 50%);
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
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
    }
    
    .main-header h1 {
        font-family: 'Orbitron', sans-serif;
        margin: 0;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: 3px;
        text-shadow: 
            0 0 10px rgba(255,255,255,0.5),
            0 0 20px rgba(13, 110, 253, 0.8);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Card styling */
    .stCard {
        background: rgba(10, 14, 39, 0.8);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(13, 110, 253, 0.3);
        box-shadow: 0 8px 32px 0 rgba(13, 110, 253, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%);
        color: white;
        border: 2px solid rgba(13, 110, 253, 0.5);
        border-radius: 12px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 20px rgba(13, 110, 253, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 30px rgba(13, 110, 253, 0.6);
        border-color: rgba(255, 255, 255, 0.6);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(10, 14, 39, 0.6);
        border: 2px dashed rgba(13, 110, 253, 0.5);
        border-radius: 15px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    /* Progress styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0d6efd 0%, #6610f2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(13, 110, 253, 0.3);
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(10, 14, 39, 0.8);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(13, 110, 253, 0.3);
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(13, 110, 253, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(13, 110, 253, 0.6);
        box-shadow: 0 6px 25px rgba(13, 110, 253, 0.4);
        transform: translateY(-2px);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 2px solid;
        box-shadow: 0 0 15px currentColor;
    }
    
    .status-complete {
        background: rgba(25, 135, 84, 0.2);
        color: #20c997;
        border-color: #20c997;
    }
    
    .status-progress {
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border-color: #ffc107;
    }
    
    .status-waiting {
        background: rgba(108, 117, 125, 0.2);
        color: #adb5bd;
        border-color: #6c757d;
    }
    
    .status-error {
        background: rgba(220, 53, 69, 0.2);
        color: #ff6b6b;
        border-color: #ff6b6b;
    }
    
    /* Text colors */
    .stMarkdown, p, span, label {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(13, 110, 253, 0.5);
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #0dcaf0;
        text-shadow: 0 0 10px rgba(13, 202, 240, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_initialized' not in st.session_state:
    st.session_state.session_initialized = True
    st.session_state.uploaded_file = None
    st.session_state.raw_data = None
    st.session_state.clean_data = None
    st.session_state.ml_results = None
    st.session_state.visualizations = None
    st.session_state.pipeline_stage = 'upload'  # upload, cleaning, ml, viz, complete
    st.session_state.cleaning_complete = False
    st.session_state.ml_complete = False
    st.session_state.viz_complete = False

# Hero Header
st.markdown("""
    <div class="main-header">
        <h1>🤖 NEURAL DATA SCIENCE CORE</h1>
        <p>Autonomous AI-Powered Data Analysis Quantum System</p>
    </div>
""", unsafe_allow_html=True)

# Welcome Section
st.markdown("## 🎯 QUANTUM DATA INTELLIGENCE PLATFORM")
st.markdown("""
This next-generation AI system automates your complete data science workflow:
- **⚡ Neural Data Purification** - AI semantic understanding & quantum cleaning
- **🧠 Intelligent ML Core** - Autonomous model selection & optimization
- **🎨 Dynamic Visualizations** - Smart PowerBI-style dashboards with AI recommendations

**Upload your data and let the neural network do the rest!**
""")

st.divider()

    # File Upload Section
st.markdown("## 📂 Upload Your Data")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to begin the automated analysis"
    )
    
    if uploaded_file is not None:
        if uploaded_file != st.session_state.uploaded_file:
            # New file uploaded
            st.session_state.uploaded_file = uploaded_file
            st.session_state.pipeline_stage = 'uploaded'
            
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.raw_data = df
            st.success(f"✅ File uploaded: **{uploaded_file.name}**")
            st.info(f"📊 Dataset: {len(df):,} rows × {len(df.columns)} columns")

with col2:
    if st.session_state.raw_data is not None:
        st.metric("Rows", f"{len(st.session_state.raw_data):,}")
        st.metric("Columns", len(st.session_state.raw_data.columns))

with col3:
    if st.session_state.raw_data is not None:
        missing = st.session_state.raw_data.isnull().sum().sum()
        missing_pct = (missing / (len(st.session_state.raw_data) * len(st.session_state.raw_data.columns))) * 100
        st.metric("Missing", f"{missing_pct:.1f}%")
        st.metric("Memory", f"{st.session_state.raw_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Pipeline Status
if st.session_state.raw_data is not None:
    st.divider()
    st.markdown("## 🔄 Pipeline Status")
    
    # Progress indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.cleaning_complete:
            st.markdown("✅ **Data Cleaning**")
            st.markdown('<span class="status-badge status-complete">COMPLETE</span>', unsafe_allow_html=True)
        elif st.session_state.pipeline_stage == 'cleaning':
            st.markdown("🔄 **Data Cleaning**")
            st.markdown('<span class="status-badge status-progress">IN PROGRESS</span>', unsafe_allow_html=True)
        else:
            st.markdown("⏳ **Data Cleaning**")
            st.markdown('<span class="status-badge status-waiting">WAITING</span>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.ml_complete:
            st.markdown("✅ **ML Training**")
            st.markdown('<span class="status-badge status-complete">COMPLETE</span>', unsafe_allow_html=True)
        elif st.session_state.pipeline_stage == 'ml':
            st.markdown("🔄 **ML Training**")
            st.markdown('<span class="status-badge status-progress">IN PROGRESS</span>', unsafe_allow_html=True)
        else:
            st.markdown("⏳ **ML Training**")
            st.markdown('<span class="status-badge status-waiting">WAITING</span>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.viz_complete:
            st.markdown("✅ **Visualizations**")
            st.markdown('<span class="status-badge status-complete">COMPLETE</span>', unsafe_allow_html=True)
        elif st.session_state.pipeline_stage == 'viz':
            st.markdown("🔄 **Visualizations**")
            st.markdown('<span class="status-badge status-progress">IN PROGRESS</span>', unsafe_allow_html=True)
        else:
            st.markdown("⏳ **Visualizations**")
            st.markdown('<span class="status-badge status-waiting">WAITING</span>', unsafe_allow_html=True)
    
    # Overall progress bar
    total_steps = 3
    completed_steps = sum([
        st.session_state.cleaning_complete,
        st.session_state.ml_complete,
        st.session_state.viz_complete
    ])
    progress = completed_steps / total_steps
    
    st.progress(progress)
    st.markdown(f"**Overall Progress:** {completed_steps}/{total_steps} steps complete ({progress*100:.0f}%)")
    
    st.divider()
    
    # Next steps
    st.markdown("## 🚀 Next Steps")
    
    if not st.session_state.cleaning_complete:
        st.info("👉 Navigate to **🧹 Data Cleaning** in the sidebar to start the pipeline")
        if st.button("🚀 Start Data Cleaning", type="primary"):
            st.switch_page("pages/1_🧹_Data_Cleaning.py")
    
    elif not st.session_state.ml_complete:
        st.info("👉 Navigate to **🧠 ML Training** in the sidebar to train models")
        if st.button("🚀 Start ML Training", type="primary"):
            st.switch_page("pages/2_🧠_ML_Training.py")
    
    elif not st.session_state.viz_complete:
        st.info("👉 Navigate to **🎨 Visualization** in the sidebar to generate dashboards")
        if st.button("🚀 Generate Visualizations", type="primary"):
            st.switch_page("pages/3_🎨_Visualization.py")
    
    else:
        st.success("🎉 **Analysis Complete!** All pipeline stages finished successfully.")
        st.balloons()
        
        if st.button("� Start New Analysis", type="primary", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

else:
    # Show getting started guide
    st.divider()
    st.markdown("## 🎯 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 What You Need
        - A CSV or Excel file with your data
        - At least 50 rows for meaningful analysis
        - Column headers in the first row
        
        ### 🎯 What You'll Get
        - Cleaned and validated dataset
        - Trained ML models with predictions
        - Beautiful visualizations
        - Business insights in plain English
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Quick Tips
        - The agent learns from experience
        - Larger datasets = better insights
        - All steps are automated after upload
        - You can override AI recommendations
        
        ### 💡 Example Use Cases
        - Sales forecasting
        - Customer churn prediction
        - Price optimization
        - Trend analysis
        """)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    st.markdown("Use the pages above to navigate through the pipeline")
    
    st.divider()
    
    st.markdown("## 📊 Session Info")
    if st.session_state.raw_data is not None:
        st.markdown(f"**File:** {st.session_state.uploaded_file.name}")
        st.markdown(f"**Rows:** {len(st.session_state.raw_data):,}")
        st.markdown(f"**Columns:** {len(st.session_state.raw_data.columns)}")
    else:
        st.markdown("*No data uploaded yet*")
    
    st.divider()
    
    st.markdown("## ⚙️ Quick Actions")
    if st.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    st.markdown("## 📚 Resources")
    st.markdown("[📖 Documentation](https://github.com/SelimRiahi/datascience-agent)")
    st.markdown("[🐛 Report Issue](https://github.com/SelimRiahi/datascience-agent/issues)")
    st.markdown("[⭐ Star on GitHub](https://github.com/SelimRiahi/datascience-agent)")
    
    st.divider()
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Selim Riahi")
