"""
🎨 AI-POWERED POWERBI-STYLE DASHBOARD GENERATOR
Terminal-based workflow with interactive filters and slicers
Now with SMART LEARNING - uses database when confident, LLM when needed!
"""

import os
import sys
import pandas as pd
import json
from smart_recommender import SmartRecommender

def main():
    print("\n" + "="*60)
    print("🎨 AI-POWERED POWERBI DASHBOARD GENERATOR")
    print("🧠 WITH SMART LEARNING SYSTEM")
    print("="*60 + "\n")
    
    # Initialize smart recommender (includes learning database)
    smart_recommender = SmartRecommender(confidence_threshold=0.75)
    
    print("🧠 Learning Database Status:")
    stats = smart_recommender.db.get_database_stats()
    print(f"   📊 Total sessions recorded: {stats['total_sessions']}")
    print(f"   🤖 LLM recommendations: {stats['llm_recommendations']}")
    print(f"   💾 Database recommendations: {stats['database_recommendations']}")
    
    if stats['total_recommendations'] > 0:
        db_pct = (stats['database_recommendations'] / stats['total_recommendations']) * 100
        print(f"   ⚡ Efficiency: {db_pct:.1f}% from database (saves money!)")
    
    print(f"   📚 Learned patterns: {stats['learned_patterns']}")
    
    # Step 1: Select dataset
    print("📁 Available datasets in ../data/:")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    choice = int(input("\nSelect dataset (number): "))
    dataset_path = os.path.join(data_dir, csv_files[choice - 1])
    dataset_name = csv_files[choice - 1]
    
    # Load dataset - auto-detect delimiter
    try:
        df = pd.read_csv(dataset_path)
        # Check if we only got 1 column with weird names (means wrong delimiter)
        if df.shape[1] == 1 and (';' in df.columns[0] or '|' in df.columns[0] or '\t' in df.columns[0]):
            # Try different delimiters
            for sep in [';', '|', '\t']:
                try:
                    df = pd.read_csv(dataset_path, sep=sep)
                    if df.shape[1] > 1:
                        print(f"   ℹ️ Auto-detected delimiter: '{sep}'")
                        break
                except:
                    continue
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    print(f"\n✅ Loaded {dataset_name}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Step 2: Select columns (with multi-group support)
    all_selected_columns = []
    column_groups = []
    
    while True:
        print(f"\n📊 Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col} ({df[col].dtype})")
        
        if len(all_selected_columns) > 0:
            print(f"\n✅ Already selected: {all_selected_columns}")
        
        col_choices = input("\n🎯 Select columns for analysis (comma-separated numbers): ")
        col_indices = [int(x.strip()) - 1 for x in col_choices.split(',')]
        selected_columns = [df.columns[i] for i in col_indices]
        
        column_groups.append(selected_columns)
        all_selected_columns.extend(selected_columns)
        
        print(f"\n✅ Group {len(column_groups)}: {selected_columns}")
        
        # Ask if user wants to add more groups
        add_more = input("\n➕ Add another group of columns? (y/n): ").strip().lower()
        if add_more != 'y':
            break
    
    print(f"\n🎯 Final selection: {len(column_groups)} group(s) with {len(all_selected_columns)} total columns")
    for i, group in enumerate(column_groups, 1):
        print(f"   Group {i}: {group}")
    
    # 🧠 RECORD SESSION IN DATABASE
    print("\n🧠 Recording session in learning database...")
    session_id = smart_recommender.db.record_session(
        dataset_name=dataset_name,
        df=df,
        selected_columns=all_selected_columns,
        group_structure=column_groups
    )
    print(f"   ✅ Session ID: {session_id}")
    
    # Step 3: Get SMART recommendations for all groups
    print("\n🤖 Getting intelligent recommendations...")
    print("   (Using database when confident, LLM when needed)")
    
    all_recommendations = []
    for i, group in enumerate(column_groups, 1):
        print(f"\n   📊 Analyzing Group {i}: {group}")
        
        # 🧠 USE SMART RECOMMENDER (Database-first approach)
        result = smart_recommender.get_recommendations(
            df=df,
            selected_columns=group,
            dataset_name=dataset_name,
            session_id=session_id
        )
        
        # Show what happened
        if result['source'] == 'database':
            print(f"      ✅ Used database (confidence: {result['confidence']*100:.0f}%)")
            print(f"      💰 Cost: $0.00 - No LLM call needed!")
        elif result['source'] == 'llm':
            print(f"      🤖 Called LLM ({result['reasoning']})")
            print(f"      💰 Cost: ${result['cost']:.2f}")
        else:  # hybrid
            print(f"      🔄 Used hybrid approach (confidence: {result['confidence']*100:.0f}%)")
            print(f"      💰 Cost: ${result['cost']:.2f}")
        
        all_recommendations.append({
            'group_id': i,
            'columns': group,
            'recommendations': result['recommendations']
        })
    
    print(f"\n✅ Analysis complete for {len(column_groups)} group(s)!")
    
    # Show efficiency stats
    session_stats = smart_recommender.get_stats()
    if session_stats['db_hits'] > 0:
        print(f"\n💡 This session efficiency:")
        print(f"   💾 Database used: {session_stats['db_hits']} time(s)")
        print(f"   🤖 LLM called: {session_stats['llm_calls']} time(s)")
        print(f"   💰 Saved: ${session_stats.get('estimated_savings', 0):.2f}")
    
    # Get the last recommendations for display
    recommendations = all_recommendations[-1]['recommendations'] if all_recommendations else {}
    print("\n📊 Sample Recommendations:")
    if 'individual_visualizations' in recommendations:
        for viz in recommendations['individual_visualizations'][:3]:
            col_name = viz.get('column', 'unknown')
            viz_type = viz.get('visualization_type') or viz.get('type', 'unknown')
            print(f"   • {col_name} → {viz_type}")
    
    # Step 4: Generate multi-page dashboard (Stats + PowerBI)
    print("\n✨ Generating multi-page dashboard (Statistics Report + PowerBI Dashboard)...")
    
    # Save temp data (all selected columns)
    temp_data = df[all_selected_columns]
    temp_data_file = "temp_data.csv"
    temp_data.to_csv(temp_data_file, index=False)
    
    # Save all recommendations with groups
    recommendations_file = "temp_recommendations.json"
    with open(recommendations_file, "w") as f:
        json.dump({
            'column_groups': column_groups,
            'all_recommendations': all_recommendations
        }, f, indent=2)
    
    # Use the beautiful dashboard
    dashboard_file = "beautiful_dashboard.py"
    
    print(f"✅ Dashboard ready: {dashboard_file}")
    print(f"\n📊 Column Groups:")
    for i, group in enumerate(column_groups, 1):
        print(f"   Group {i}: {group}")
    print("\n📄 Dashboard Pages:")
    print("   1️⃣  Statistics Report (Advanced with plots, diagrams, correlation matrix!)")
    print("   2️⃣  PowerBI Dashboard (with Anneau, Courbe, Barre 3D, etc.)")
    
    # Step 5: Launch dashboard
    print("\n🚀 Launching multi-page dashboard...")
    import subprocess
    import webbrowser
    import time
    
    # Set environment variable to prevent Streamlit from auto-opening browser
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Launch Streamlit without auto-browser
    dashboard_file = "beautiful_dashboard.py"
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", dashboard_file,
        "--server.headless", "true"
    ])
    
    # Wait and open browser ONCE
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    
    print("\n✅ Dashboard launched at http://localhost:8501")
    print("\n📄 Available Pages:")
    print("   📊 Statistics Report - Your favorite stats page")
    print("   🎛️ PowerBI Dashboard - Interactive with filters")
    print("\n🎨 PowerBI Elements:")
    print("   • 🍩 Anneau (Donut Charts)")
    print("   • 📊 Barre 3D (3D Bar Charts)")
    print("   • 📈 Courbe (Line/Curve Charts)")
    print("   • 📊 Histogramme (Histograms)")
    print("   • 📈 Zone (Area Charts)")
    print("   • 🎯 Jauge (Gauge Charts)")
    print("   • 🎛️ Filters & Slicers")
    
    # 🧠 Show final learning stats
    print("\n🧠 Learning Database Updated!")
    stats = smart_recommender.db.get_database_stats()
    print(f"   📊 Total sessions: {stats['total_sessions']}")
    print(f"   💡 Total recommendations: {stats['total_recommendations']}")
    
    if stats['total_recommendations'] > 0:
        db_pct = (stats['database_recommendations'] / stats['total_recommendations']) * 100
        print(f"   ⚡ Overall efficiency: {db_pct:.1f}% from database")
        print(f"   💰 Total saved: ${stats['database_recommendations'] * 0.05:.2f}")
    
    # Close connections
    smart_recommender.close()

def generate_powerbi_dashboard(dataset_name, selected_columns, data_file, recommendations_file):
    """Generate PowerBI-style dashboard code"""
    
    return f'''"""
📊 POWERBI-STYLE INTERACTIVE DASHBOARD
Auto-generated by AI-Powered Visualization System
Dataset: {dataset_name}
Columns: {", ".join(selected_columns)}
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page config
st.set_page_config(
    page_title="📊 PowerBI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for PowerBI-like styling
st.markdown("""
    <style>
    .main {{
        background-color: #f5f5f5;
    }}
    
    .kpi-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    
    .kpi-card-green {{
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    
    .kpi-card-orange {{
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    
    .kpi-card-blue {{
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }}
    
    .kpi-value {{
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }}
    
    .kpi-label {{
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }}
    
    h1 {{
        color: #2c3e50;
        font-weight: 700;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: #2c3e50;
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Load embedded data
df = pd.read_csv("{data_file}")
with open("{recommendations_file}", "r") as f:
    recommendations = json.load(f)

# Sidebar - Filters & Slicers (PowerBI style)
with st.sidebar:
    st.title("🎛️ FILTERS & SLICERS")
    st.markdown("---")
    
    # Dynamic filters based on column types
    filters = {{}}
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            # Categorical filter (multiselect)
            st.subheader(f"📌 {{col}}")
            unique_vals = df[col].unique().tolist()
            selected = st.multiselect(
                f"Select {{col}}",
                options=unique_vals,
                default=unique_vals,
                key=f"filter_{{col}}"
            )
            filters[col] = selected
        
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numeric filter (slider)
            st.subheader(f"🔢 {{col}}")
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            selected_range = st.slider(
                f"Range for {{col}}",
                min_val, max_val, (min_val, max_val),
                key=f"slider_{{col}}"
            )
            filters[col] = selected_range
    
    # Apply filters button
    st.markdown("---")
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()

# Apply filters to dataframe
filtered_df = df.copy()
for col, filter_val in filters.items():
    if df[col].dtype == 'object' or df[col].nunique() < 20:
        filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
    elif pd.api.types.is_numeric_dtype(df[col]):
        filtered_df = filtered_df[
            (filtered_df[col] >= filter_val[0]) & 
            (filtered_df[col] <= filter_val[1])
        ]

# Main Dashboard
st.title("📊 Interactive Analytics Dashboard")
st.markdown(f"**📊 Dataset:** {dataset_name}")
filter_pct = len(filtered_df) / len(df) * 100
st.markdown(f"**📋 Filtered Records:** {{len(filtered_df):,}} / {{len(df):,}} total ({{filter_pct:.1f}}%)")
st.markdown("---")

# KPI Cards Row (PowerBI style)
st.subheader("📈 Key Performance Indicators")

numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

if len(numeric_cols) >= 4:
    kpi_cols = st.columns(4)
    colors = ['kpi-card', 'kpi-card-green', 'kpi-card-orange', 'kpi-card-blue']
    
    for i, col in enumerate(numeric_cols[:4]):
        with kpi_cols[i]:
            avg_val = filtered_df[col].mean()
            if avg_val > 1000:
                display_val = f"{{avg_val:,.0f}}"
            else:
                display_val = f"{{avg_val:.2f}}"
            
            st.markdown(f"""
            <div class="{{colors[i]}}">
                <div class="kpi-value">{{display_val}}</div>
                <div class="kpi-label">Avg {{col}}</div>
            </div>
            """, unsafe_allow_html=True)
elif len(numeric_cols) > 0:
    kpi_cols = st.columns(len(numeric_cols))
    colors = ['kpi-card', 'kpi-card-green', 'kpi-card-orange', 'kpi-card-blue']
    
    for i, col in enumerate(numeric_cols):
        with kpi_cols[i]:
            avg_val = filtered_df[col].mean()
            if avg_val > 1000:
                display_val = f"{{avg_val:,.0f}}"
            else:
                display_val = f"{{avg_val:.2f}}"
            
            st.markdown(f"""
            <div class="{{colors[i % 4]}}">
                <div class="kpi-value">{{display_val}}</div>
                <div class="kpi-label">Avg {{col}}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Main Visualizations Grid
if recommendations:
    # Row 1: Individual visualizations
    st.subheader("📊 Individual Metrics Analysis")
    
    individual_viz = recommendations.get('individual_visualizations', [])
    
    # Create two columns for side-by-side charts
    if len(individual_viz) >= 2:
        col1, col2 = st.columns(2)
        
        for idx, viz in enumerate(individual_viz[:2]):
            col_name = viz['column']
            
            if col_name in filtered_df.columns:
                with (col1 if idx == 0 else col2):
                    st.markdown(f"**{{viz['styling']['title']}}**")
                    
                    if viz['viz_type'] == 'histogram':
                        fig = px.histogram(
                            filtered_df, x=col_name,
                            color_discrete_sequence=['#667eea'],
                            template='plotly_white'
                        )
                    elif viz['viz_type'] == 'bar_chart':
                        value_counts = filtered_df[col_name].value_counts().reset_index()
                        value_counts.columns = [col_name, 'count']
                        fig = px.bar(
                            value_counts, x=col_name, y='count',
                            color_discrete_sequence=['#11998e'],
                            template='plotly_white'
                        )
                    elif viz['viz_type'] == 'box_plot':
                        fig = px.box(
                            filtered_df, y=col_name,
                            color_discrete_sequence=['#f5576c'],
                            template='plotly_white'
                        )
                    else:
                        fig = px.histogram(filtered_df, x=col_name, template='plotly_white')
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{{idx}}_{{col_name}}")
                    
                    with st.expander("💡 AI Insight"):
                        st.info(viz['reasoning'])
    
    # Show any remaining individual visualizations
    if len(individual_viz) > 2:
        for idx, viz in enumerate(individual_viz[2:], start=2):
            col_name = viz['column']
            
            if col_name in filtered_df.columns:
                st.markdown(f"**{{viz['styling']['title']}}**")
                
                if viz['viz_type'] == 'histogram':
                    fig = px.histogram(filtered_df, x=col_name, template='plotly_white')
                elif viz['viz_type'] == 'bar_chart':
                    value_counts = filtered_df[col_name].value_counts().reset_index()
                    value_counts.columns = [col_name, 'count']
                    fig = px.bar(value_counts, x=col_name, y='count', template='plotly_white')
                elif viz['viz_type'] == 'box_plot':
                    fig = px.box(filtered_df, y=col_name, template='plotly_white')
                else:
                    fig = px.histogram(filtered_df, x=col_name, template='plotly_white')
                
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{{idx}}_{{col_name}}")
                
                with st.expander("💡 AI Insight"):
                    st.info(viz['reasoning'])
    
    st.markdown("---")
    
    # Row 2: Relationship visualizations (full width)
    relationship_viz = recommendations.get('relationship_visualizations', [])
    if relationship_viz:
        st.subheader("🔗 Relationships & Correlations")
        
        for idx, viz in enumerate(relationship_viz):
            cols = viz['columns']
            
            if all(col in filtered_df.columns for col in cols):
                st.markdown(f"**{{viz['styling']['title']}}**")
                
                if viz['viz_type'] == 'scatter':
                    fig = px.scatter(
                        filtered_df, x=cols[0], y=cols[1],
                        template='plotly_white',
                        opacity=0.6,
                        trendline="ols",
                        color_discrete_sequence=['#667eea']
                    )
                elif viz['viz_type'] == 'grouped_bar':
                    # Determine categorical and numeric
                    if filtered_df[cols[0]].dtype == 'object' or filtered_df[cols[0]].nunique() < 20:
                        cat_col, num_col = cols[0], cols[1]
                    else:
                        cat_col, num_col = cols[1], cols[0]
                    
                    fig = px.box(
                        filtered_df, x=cat_col, y=num_col,
                        color=cat_col,
                        template='plotly_white'
                    )
                else:
                    fig = px.scatter(filtered_df, x=cols[0], y=cols[1], template='plotly_white')
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"rel_chart_{{idx}}")
                
                with st.expander("💡 AI Insight"):
                    st.info(viz['reasoning'])
                
                st.markdown("---")

# Row 3: Data table with highlighting
st.subheader("📋 Detailed Data View")

if len(numeric_cols) > 0:
    # Show styled dataframe with conditional formatting
    st.dataframe(
        filtered_df.style.background_gradient(
            subset=[numeric_cols[0]],
            cmap='RdYlGn'
        ),
        use_container_width=True,
        height=400
    )
else:
    st.dataframe(filtered_df, use_container_width=True, height=400)

# Download filtered data
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv",
    use_container_width=True
)

# Footer
st.markdown("---")
st.markdown("*🤖 Generated by AI-Powered Visualization System*")
'''

if __name__ == "__main__":
    main()
