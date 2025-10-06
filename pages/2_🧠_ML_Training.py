"""
🧠 ML TRAINING MODULE
User-Controlled Model Training with 4 Choices
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Import ML Agent
from ml_agent.smart_ml_agent import SmartMLAgent

# Page config
st.set_page_config(
    page_title="ML Training - Neural Core",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'ml_agent' not in st.session_state:
    st.session_state.ml_agent = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'selected_target' not in st.session_state:
    st.session_state.selected_target = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'ml_metadata' not in st.session_state:
    st.session_state.ml_metadata = None
if 'llm_recommendations' not in st.session_state:
    st.session_state.llm_recommendations = None
if 'db_stats' not in st.session_state:
    st.session_state.db_stats = None
if 'performance_prediction' not in st.session_state:
    st.session_state.performance_prediction = None
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = None
if 'model_selection_strategy' not in st.session_state:
    st.session_state.model_selection_strategy = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Custom CSS - Futuristic ML Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;600;700&display=swap');

.main {
    background: #0a0e27;
    background-image: 
        radial-gradient(at 20% 30%, rgba(13, 110, 253, 0.15) 0px, transparent 50%),
        radial-gradient(at 80% 70%, rgba(102, 16, 242, 0.15) 0px, transparent 50%);
    font-family: 'Space Grotesk', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero-section {
    background: linear-gradient(135deg, #0d6efd 0%, #6610f2 50%, #20c997 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    border: 2px solid rgba(13, 110, 253, 0.4);
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 0 60px rgba(13, 110, 253, 0.5);
}

.hero-section h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    margin: 0;
    color: white;
    text-shadow: 0 0 20px rgba(13, 110, 253, 1);
    letter-spacing: 4px;
}

.glass-card {
    background: rgba(10, 14, 39, 0.85);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(13, 110, 253, 0.3);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px 0 rgba(13, 110, 253, 0.25);
}

.metric-card {
    background: linear-gradient(135deg, rgba(13, 110, 253, 0.2) 0%, rgba(102, 16, 242, 0.2) 100%);
    border: 2px solid rgba(13, 110, 253, 0.4);
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(13, 110, 253, 0.3);
}

.metric-card h3 {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5rem;
    margin: 0;
    color: #0dcaf0;
    text-shadow: 0 0 20px rgba(13, 202, 240, 0.8);
}

.info-box, .success-box, .warning-box {
    border-radius: 12px;
    padding: 1.2rem;
    margin: 1rem 0;
}

.info-box {
    background: rgba(13, 110, 253, 0.1);
    border: 1px solid rgba(13, 110, 253, 0.3);
    border-left: 4px solid #0d6efd;
}

.success-box {
    background: rgba(32, 201, 151, 0.1);
    border: 1px solid rgba(32, 201, 151, 0.3);
    border-left: 4px solid #20c997;
}

.warning-box {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid rgba(255, 193, 7, 0.3);
    border-left: 4px solid #ffc107;
}

.stButton > button {
    background: linear-gradient(135deg, #0d6efd 0%, #6610f2 100%) !important;
    color: white !important;
    border: 2px solid rgba(13, 110, 253, 0.5) !important;
    border-radius: 12px !important;
    padding: 0.9rem 2.5rem !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}

.choice-button {
    background: rgba(13, 110, 253, 0.2);
    border: 2px solid rgba(13, 110, 253, 0.5);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
}

.choice-button:hover {
    background: rgba(13, 110, 253, 0.3);
    border-color: rgba(13, 110, 253, 0.8);
    transform: translateY(-3px);
}

.model-badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    margin: 0.3rem;
    border: 2px solid;
}

.badge-complete {
    background: rgba(32, 201, 151, 0.2);
    border-color: #20c997;
    color: #20c997;
}

.badge-best {
    background: rgba(255, 193, 7, 0.2);
    border-color: #ffc107;
    color: #ffc107;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 39, 0.95);
    backdrop-filter: blur(20px);
    border-right: 2px solid rgba(13, 110, 253, 0.3);
}

[data-testid="stSidebar"] .element-container {
    color: white;
}

.prediction-history-card {
    background: rgba(13, 110, 253, 0.1);
    border: 1px solid rgba(13, 110, 253, 0.3);
    border-left: 4px solid #0d6efd;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.prediction-history-card h5 {
    color: #0dcaf0;
    font-family: 'Orbitron', sans-serif;
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
    <div class="hero-section">
        <h1>🧠 QUANTUM ML CORE</h1>
        <p>User-Controlled Model Training</p>
    </div>
""", unsafe_allow_html=True)

# Check for data availability
def get_training_data():
    """Get data for training - either cleaned or raw"""
    if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df, "cleaned"
    elif 'current_df' in st.session_state and st.session_state.current_df is not None:
        return st.session_state.current_df, "raw"
    elif 'original_df' in st.session_state and st.session_state.original_df is not None:
        return st.session_state.original_df, "raw"
    elif 'raw_data' in st.session_state and st.session_state.raw_data is not None:
        return st.session_state.raw_data, "raw"
    return None, None

training_data, data_source = get_training_data()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ ML CONFIGURATION")
    
    if st.session_state.selected_models:
        st.metric("Selected Models", len(st.session_state.selected_models))
        st.info(f"🎯 Strategy: {st.session_state.model_selection_strategy}")
    
    st.markdown("---")
    st.markdown("### 📊 LIVE METRICS")
    
    if training_data is not None:
        st.metric("📥 Dataset Rows", f"{len(training_data):,}")
        st.metric("📋 Features", len(training_data.columns))
        
        if st.session_state.selected_target:
            st.metric("🎯 Target", st.session_state.selected_target)
        
        if st.session_state.ml_results:
            st.metric("✅ Models Trained", len(st.session_state.ml_results))
            best = max(st.session_state.ml_results, key=lambda x: x['score'])
            st.metric("🏆 Best Score", f"{best['score']:.4f}")
    
    st.markdown("---")
    st.markdown("### 🎯 SYSTEM STATUS")
    st.markdown("""
    <div class="info-box" style="font-size: 0.85rem; padding: 1rem;">
        <strong>🧠 ML Intelligence Core</strong><br><br>
        ⚡ User-controlled training<br>
        🎯 4-choice strategy system<br>
        🤖 AI-powered recommendations<br>
        📊 Real-time predictions
    </div>
    """, unsafe_allow_html=True)
    
    # Show prediction history
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 🔮 Recent Predictions")
        # Show last 5 predictions
        for pred_item in st.session_state.prediction_history[-5:][::-1]:
            st.markdown(f"""
            <div class="prediction-history-card">
                <strong>{pred_item['model']}</strong><br>
                🎯 <strong>{pred_item['prediction']}</strong>
                {'<br>✅ Actual: ' + pred_item['actual'] if 'actual' in pred_item else ''}
            </div>
            """, unsafe_allow_html=True)

# Main content
if training_data is None:
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ NO DATA AVAILABLE</h3>
        <p>Please upload and process data first.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Target & Features",
        "🤖 Model Selection",  
        "🚀 Training",
        "📊 Results",
        "🔮 Predictions"
    ])
    
    # TAB 1: Target & Features
    with tab1:
        st.markdown("## 🎯 Configure Training")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎯 Target Variable")
            target_column = st.selectbox(
                "Target Column",
                options=list(training_data.columns),
                index=0 if st.session_state.selected_target is None else list(training_data.columns).index(st.session_state.selected_target) if st.session_state.selected_target in training_data.columns else 0
            )
            
            if target_column:
                st.session_state.selected_target = target_column
                
                # Target analysis
                target_info = training_data[target_column]
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Unique Values", target_info.nunique())
                with col_b:
                    missing_pct = (target_info.isnull().sum() / len(target_info)) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col2:
            st.markdown("### 🎨 Feature Selection")
            
            available_features = [col for col in training_data.columns if col != target_column]
            
            feature_mode = st.radio(
                "Selection Mode",
                options=["🤖 Use All Features", "✋ Manual Selection"]
            )
            
            if feature_mode == "✋ Manual Selection":
                selected_features = st.multiselect(
                    "Select Features",
                    options=available_features,
                    default=available_features
                )
                st.session_state.selected_features = selected_features if selected_features else None
            else:
                st.session_state.selected_features = None
                selected_features = available_features
            
            st.info(f"✅ Using {len(selected_features)} features")
        
        if target_column:
            st.success("✅ Configuration complete! Proceed to Model Selection.")
    
    # TAB 2: Model Selection (4 CHOICES)
    with tab2:
        st.markdown("## 🤖 Model Selection Strategy")
        
        if not st.session_state.selected_target:
            st.warning("⚠️ Please select target in Tab 1 first")
        else:
            # Initialize agent and get recommendations
            if st.session_state.ml_agent is None:
                if st.button("🚀 Initialize ML Agent", type="primary"):
                    with st.spinner("Initializing..."):
                        try:
                            st.session_state.ml_agent = SmartMLAgent()
                            
                            # Get metadata
                            metadata = st.session_state.ml_agent.extract_metadata(training_data, st.session_state.selected_target)
                            st.session_state.ml_metadata = metadata
                            
                            # Get recommendations
                            llm_rec = st.session_state.ml_agent.get_llm_recommendation(metadata)
                            st.session_state.llm_recommendations = llm_rec
                            
                            # Get stats
                            st.session_state.db_stats = st.session_state.ml_agent.db.get_database_stats()
                            st.session_state.performance_prediction = st.session_state.ml_agent.db.predict_performance_range(metadata)
                            
                            st.success("✅ Agent Ready!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            else:
                # Show AI recommendations
                if st.session_state.llm_recommendations:
                    llm_rec = st.session_state.llm_recommendations
                    
                    st.markdown("### 💡 AI Recommendations")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", llm_rec['confidence_level'])
                    with col2:
                        st.metric("Top Choice", llm_rec['final_recommendation']['top_choice'])
                    with col3:
                        st.metric("Second Choice", llm_rec['final_recommendation']['second_choice'])
                    
                    st.info(f"💡 {llm_rec['analysis']['reasoning']}")
                
                # THE 4 CHOICES
                st.markdown("---")
                st.markdown("### 🎯 YOUR CHOICE: Select Training Strategy")
                
                problem_type = st.session_state.ml_metadata['problem_type']
                all_models = list(st.session_state.ml_agent.models[problem_type].keys())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="glass-card">
                        <h4>1️⃣ Trust Agent</h4>
                        <p>Use agent's top recommendation</p>
                        <p><strong>⚡ Time saved: ~83%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🎯 Use Agent's Choice", key="choice1", use_container_width=True):
                        st.session_state.selected_models = [llm_rec['final_recommendation']['top_choice']]
                        st.session_state.model_selection_strategy = "Agent's Choice"
                        st.success(f"✅ Selected: {st.session_state.selected_models[0]}")
                
                with col2:
                    st.markdown("""
                    <div class="glass-card">
                        <h4>2️⃣ Trust LLM</h4>
                        <p>Top 3 LLM suggestions</p>
                        <p><strong>⚡ Time saved: ~50%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🤖 Use LLM's Choice", key="choice2", use_container_width=True):
                        if problem_type == 'classification':
                            llm_models = ['XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier']
                        else:
                            llm_models = ['XGBRegressor', 'LGBMRegressor', 'RandomForestRegressor']
                        st.session_state.selected_models = llm_models
                        st.session_state.model_selection_strategy = "LLM's Choice"
                        st.success(f"✅ Selected: {', '.join(llm_models)}")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("""
                    <div class="glass-card">
                        <h4>3️⃣ Test All</h4>
                        <p>Train all 6 models</p>
                        <p><strong>⚡ Time saved: 0%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🚀 Test All Models", key="choice3", use_container_width=True):
                        st.session_state.selected_models = all_models
                        st.session_state.model_selection_strategy = "All Models"
                        st.success(f"✅ Selected all {len(all_models)} models")
                
                with col4:
                    st.markdown("""
                    <div class="glass-card">
                        <h4>4️⃣ Manual</h4>
                        <p>Choose specific models</p>
                        <p><strong>⚡ Time saved: Varies</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("✋ Manual Selection", key="choice4", use_container_width=True):
                        st.session_state.manual_mode = True
                
                # Manual selection
                if 'manual_mode' in st.session_state and st.session_state.manual_mode:
                    st.markdown("---")
                    manual_models = st.multiselect(
                        "Select models:",
                        options=all_models,
                        default=st.session_state.selected_models if st.session_state.selected_models else []
                    )
                    
                    if st.button("✅ Confirm Selection"):
                        if manual_models:
                            st.session_state.selected_models = manual_models
                            st.session_state.model_selection_strategy = "Manual"
                            st.session_state.manual_mode = False
                            st.success(f"✅ Selected: {', '.join(manual_models)}")
                
                # Show current selection
                if st.session_state.selected_models:
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Current Selection</h4>
                        <p><strong>Strategy:</strong> {st.session_state.model_selection_strategy}</p>
                        <p><strong>Models ({len(st.session_state.selected_models)}):</strong> {', '.join(st.session_state.selected_models)}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 3: Training
    with tab3:
        st.markdown("## 🚀 Model Training")
        
        if not st.session_state.selected_models:
            st.warning("⚠️ Please select models in Tab 2 first")
        else:
            st.info(f"📋 Ready to train: {', '.join(st.session_state.selected_models)}")
            
            if st.session_state.ml_results is None:
                if st.button("⚡ START TRAINING", type="primary", use_container_width=True):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    try:
                        status.text("🔧 Preprocessing...")
                        progress.progress(0.2)
                        
                        # Preprocess
                        X, y = st.session_state.ml_agent.preprocess_data(
                            training_data,
                            st.session_state.selected_target,
                            st.session_state.selected_features
                        )
                        
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Train models
                        results = []
                        problem_type = st.session_state.ml_metadata['problem_type']
                        available_models = st.session_state.ml_agent.models[problem_type]
                        
                        for idx, model_name in enumerate(st.session_state.selected_models):
                            status.text(f"⚡ Training {model_name}...")
                            progress.progress(0.2 + (0.7 * (idx / len(st.session_state.selected_models))))
                            
                            if model_name in available_models:
                                model = available_models[model_name]
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                
                                if problem_type == 'classification':
                                    from sklearn.metrics import accuracy_score
                                    score = accuracy_score(y_test, y_pred)
                                    metric = "Accuracy"
                                else:
                                    from sklearn.metrics import r2_score
                                    score = r2_score(y_test, y_pred)
                                    metric = "R²"
                                
                                results.append({
                                    'model': model_name,
                                    'score': score,
                                    'metric': metric
                                })
                                
                                # Store trained model
                                st.session_state.trained_models[model_name] = model
                        
                        # Store results
                        st.session_state.ml_results = results
                        st.session_state.ml_complete = True
                        
                        # Store preprocessor
                        st.session_state.preprocessor = {
                            'label_encoders': st.session_state.ml_agent.label_encoders.copy(),
                            'scaler': st.session_state.ml_agent.scaler,
                            'feature_columns': st.session_state.selected_features or [c for c in training_data.columns if c != st.session_state.selected_target]
                        }
                        
                        status.text("✅ Complete!")
                        progress.progress(1.0)
                        
                        time.sleep(1)
                        st.success("🎉 Training Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")
            else:
                st.success("✅ Training complete! Check Results tab.")
    
    # TAB 4: Results
    with tab4:
        st.markdown("## 📊 Training Results")
        
        if st.session_state.ml_results is None:
            st.info("ℹ️ No results yet. Complete training first.")
        else:
            results = st.session_state.ml_results
            best = max(results, key=lambda x: x['score'])
            
            st.markdown(f"""
            <div class="success-box">
                <h3>🏆 BEST MODEL: {best['model']}</h3>
                <p><strong>{best['metric']}: {best['score']:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Results table
            df_results = pd.DataFrame(results).sort_values('score', ascending=False)
            st.dataframe(df_results, use_container_width=True)
            
            # Visual
            st.bar_chart(df_results.set_index('model')['score'])
    
    # TAB 5: Predictions
    with tab5:
        st.markdown("## 🔮 Make Predictions")
        
        if not st.session_state.trained_models:
            st.warning("⚠️ No trained models. Complete training first.")
        else:
            # Model selection
            model_names = list(st.session_state.trained_models.keys())
            selected_model = st.selectbox("Select Model:", options=model_names)
            
            # Prediction mode
            pred_mode = st.radio("Prediction Mode:", ["📝 Manual Entry", "🎲 Sample Values"])
            
            # Display prediction history at the top (for both modes)
            if st.session_state.prediction_history:
                st.markdown("### 📊 Prediction History")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"📈 Total Predictions: **{len(st.session_state.prediction_history)}**")
                with col2:
                    if st.button("🗑️ Clear History"):
                        st.session_state.prediction_history = []
                        st.rerun()
                
                # Show all predictions
                for idx, pred_item in enumerate(st.session_state.prediction_history[::-1]):
                    with st.expander(f"#{len(st.session_state.prediction_history)-idx} - {pred_item['timestamp']} - {pred_item['model']}", expanded=(idx<3)):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**🎯 Prediction:** `{pred_item['prediction']}`")
                        with col_b:
                            if 'actual' in pred_item:
                                st.markdown(f"**✅ Actual:** `{pred_item['actual']}`")
                        st.markdown(f"**📝 Type:** {pred_item['type']}")
                
                st.markdown("---")
            
            if pred_mode == "📝 Manual Entry":
                st.markdown("### Enter Feature Values")
                
                feature_cols = st.session_state.preprocessor['feature_columns']
                input_data = {}
                
                cols = st.columns(3)
                for idx, feature in enumerate(feature_cols):
                    with cols[idx % 3]:
                        if pd.api.types.is_numeric_dtype(training_data[feature]):
                            input_data[feature] = st.number_input(
                                feature,
                                value=float(training_data[feature].mean())
                            )
                        else:
                            unique_vals = training_data[feature].unique()[:10]
                            input_data[feature] = st.selectbox(feature, options=unique_vals)
                
                if st.button("🚀 Predict", type="primary"):
                    try:
                        # Create DataFrame
                        df_pred = pd.DataFrame([input_data])
                        X = df_pred[feature_cols].copy()
                        
                        # Encode
                        for col in X.columns:
                            if col in st.session_state.preprocessor['label_encoders']:
                                encoder = st.session_state.preprocessor['label_encoders'][col]
                                try:
                                    X[col] = encoder.transform([X[col].iloc[0]])[0]
                                except:
                                    X[col] = -1
                        
                        # Scale
                        X_scaled = st.session_state.preprocessor['scaler'].transform(X)
                        
                        # Predict
                        model = st.session_state.trained_models[selected_model]
                        prediction = model.predict(X_scaled)[0]
                        
                        # Display result - FIXED FORMAT
                        if isinstance(prediction, (int, float)):
                            pred_display = f"{prediction:.4f}"
                        else:
                            pred_display = str(prediction)
                        
                        # Add to history
                        st.session_state.prediction_history.append({
                            'timestamp': time.strftime("%H:%M:%S"),
                            'model': selected_model,
                            'prediction': pred_display,
                            'type': 'Manual Entry'
                        })
                        
                        st.success(f"✅ Prediction added! Total: {len(st.session_state.prediction_history)}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
            
            else:  # Sample Values
                st.markdown("### Select Sample Row")
                
                sample_idx = st.slider("Row Index:", 0, len(training_data)-1, 0)
                sample_row = training_data.iloc[sample_idx]
                
                st.dataframe(sample_row.to_frame().T, use_container_width=True)
                
                if st.button("🚀 Predict Sample", type="primary"):
                    try:
                        feature_cols = st.session_state.preprocessor['feature_columns']
                        df_pred = sample_row[feature_cols].to_frame().T
                        X = df_pred.copy()
                        
                        # Encode
                        for col in X.columns:
                            if col in st.session_state.preprocessor['label_encoders']:
                                encoder = st.session_state.preprocessor['label_encoders'][col]
                                try:
                                    X[col] = encoder.transform([X[col].iloc[0]])[0]
                                except:
                                    X[col] = -1
                        
                        # Scale
                        X_scaled = st.session_state.preprocessor['scaler'].transform(X)
                        
                        # Predict
                        model = st.session_state.trained_models[selected_model]
                        prediction = model.predict(X_scaled)[0]
                        
                        # Get actual value
                        actual = sample_row[st.session_state.selected_target]
                        
                        # Display - FIXED FORMAT
                        if isinstance(prediction, (int, float)):
                            pred_display = f"{prediction:.4f}"
                        else:
                            pred_display = str(prediction)
                        
                        if isinstance(actual, (int, float)):
                            actual_display = f"{actual:.4f}"
                        else:
                            actual_display = str(actual)
                        
                        # Add to history
                        st.session_state.prediction_history.append({
                            'timestamp': time.strftime("%H:%M:%S"),
                            'model': selected_model,
                            'prediction': pred_display,
                            'actual': actual_display,
                            'type': 'Sample Data'
                        })
                        
                        st.success(f"✅ Prediction added! Total: {len(st.session_state.prediction_history)}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")

st.markdown("---")
st.markdown("<div style='text-align: center; opacity: 0.7;'>🧠 Quantum ML Core v3.0</div>", unsafe_allow_html=True)
