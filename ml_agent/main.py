"""
🤖 INTELLIGENT DATA SCIENCE AGENT
Clean User-Controlled Interface

User has FULL control:
- Choose dataset manually
- Choose target column manually  
- Choose features manually
- Choose models manually
- Test predictions manually
"""
import pandas as pd
import os
from smart_ml_agent import SmartMLAgent

class UserControlledMLAgent(SmartMLAgent):
    
    def __init__(self):
        super().__init__()
        self.data_folder = "../data"
        self.trained_models = {}  # Store trained models for predictions
        self.last_preprocessor = None  # Store preprocessing info
    
    def list_available_datasets(self):
        """List all CSV files in the data folder"""
        if not os.path.exists(self.data_folder):
            return []
        
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        return csv_files
    
    def load_and_preview_dataset(self, filename):
        """Load dataset and show preview"""
        filepath = os.path.join(self.data_folder, filename)
        
        try:
            df = pd.read_csv(filepath)
            print(f"📊 Dataset loaded: {filename}")
            print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            print(f"\n📋 Available columns:")
            for i, col in enumerate(df.columns, 1):
                dtype = df[col].dtype
                non_null = df[col].count()
                unique_vals = df[col].nunique()
                print(f"   {i:2d}. {col} ({dtype}, {non_null}/{len(df)} non-null, {unique_vals} unique)")
            
            print(f"\n👀 First 5 rows:")
            print(df.head(5).to_string())
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def get_user_target_selection(self, df):
        """Get target column from user (NO AUTO-DETECTION)"""
        columns = list(df.columns)
        
        print(f"\n🎯 SELECT TARGET COLUMN (what you want to predict):")
        for i, col in enumerate(columns, 1):
            print(f"   {i}. {col}")
        
        while True:
            try:
                target_idx = int(input(f"\nWhich column to predict? (1-{len(columns)}): ")) - 1
                if 0 <= target_idx < len(columns):
                    target_column = columns[target_idx]
                    print(f"✅ Target selected: {target_column}")
                    return target_column
                else:
                    print(f"❌ Please enter a number between 1 and {len(columns)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def get_user_feature_selection(self, df, target_column):
        """Get feature columns from user (MANUAL CONTROL)"""
        available_features = [col for col in df.columns if col != target_column]
        
        print(f"\n🔧 SELECT FEATURES (what to use for prediction):")
        print(f"Available features ({len(available_features)}):")
        for i, col in enumerate(available_features, 1):
            dtype_info = f"({df[col].dtype})"
            print(f"   {i}. {col} {dtype_info}")
        
        print(f"\n💡 Your choices:")
        print(f"   A. Use ALL {len(available_features)} features")
        print(f"   B. Select specific features manually")
        
        while True:
            choice = input(f"\nYour choice (A/B): ").strip().upper()
            
            if choice == 'A':
                selected_features = available_features
                print(f"✅ Using all {len(selected_features)} features")
                return selected_features
                
            elif choice == 'B':
                print(f"\nEnter feature numbers (comma-separated, e.g. 1,3,5):")
                while True:
                    try:
                        feature_input = input("Features: ").strip()
                        if not feature_input:
                            print("❌ Please enter at least one feature")
                            continue
                        
                        indices = [int(x.strip()) - 1 for x in feature_input.split(',')]
                        if all(0 <= idx < len(available_features) for idx in indices):
                            selected_features = [available_features[idx] for idx in indices]
                            print(f"✅ Selected {len(selected_features)} features: {selected_features}")
                            return selected_features
                        else:
                            print(f"❌ Invalid numbers. Use 1-{len(available_features)}")
                    except ValueError:
                        print("❌ Invalid format. Use comma-separated numbers")
            else:
                print("❌ Please enter A or B")
    
    def get_user_model_choice(self, metadata, enhanced_rec):
        """Let user choose which models to train"""
        available_models = list(self.models[metadata['problem_type']].keys())
        
        print(f"\n🤖 MODEL SELECTION:")
        print(f"Available {metadata['problem_type']} models:")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
        
        # Show AI recommendations but let user decide
        print(f"\n💡 AI RECOMMENDATIONS (you can ignore these):")
        print(f"   🧠 Agent suggests: {enhanced_rec['final_recommendation']['top_choice']}")
        print(f"   🤖 LLM suggests: XGBRegressor, LGBMRegressor, RandomForestRegressor")
        print(f"   📝 Agent's reasoning: {enhanced_rec['analysis']['reasoning']}")
        
        print(f"\n🎯 YOUR MODEL CHOICES:")
        print(f"   1️⃣  Trust Agent: Use only {enhanced_rec['final_recommendation']['top_choice']}")
        print(f"   2️⃣  Trust LLM: Use top 3 LLM suggestions")
        print(f"   3️⃣  Test All: Train all {len(available_models)} models")
        print(f"   4️⃣  Manual: Choose specific models yourself")
        
        while True:
            try:
                choice = input(f"\n🤔 Your choice (1-4): ").strip()
                
                if choice == '1':
                    return [enhanced_rec['final_recommendation']['top_choice']], "Agent's choice"
                    
                elif choice == '2':
                    llm_models = ['XGBRegressor', 'LGBMRegressor', 'RandomForestRegressor']
                    if metadata['problem_type'] == 'classification':
                        llm_models = ['XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier']
                    return llm_models, "LLM's choice"
                    
                elif choice == '3':
                    return available_models, "All models"
                    
                elif choice == '4':
                    print(f"\nSelect models (comma-separated numbers, e.g. 1,3,5):")
                    while True:
                        try:
                            model_input = input("Models: ").strip()
                            indices = [int(x.strip()) - 1 for x in model_input.split(',')]
                            if all(0 <= idx < len(available_models) for idx in indices):
                                selected_models = [available_models[idx] for idx in indices]
                                return selected_models, f"Manual: {selected_models}"
                            else:
                                print(f"❌ Invalid numbers. Use 1-{len(available_models)}")
                        except ValueError:
                            print("❌ Invalid format. Use comma-separated numbers")
                else:
                    print("❌ Please enter 1, 2, 3, or 4")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return None, None
    
    def train_selected_models(self, df, target_column, feature_columns, selected_models, file_path, metadata):
        """Train only the user-selected models"""
        
        # Store dataset info
        dataset_id = self.db.store_dataset_metadata(df, target_column, file_path)
        
        # Preprocess data
        X, y = self.preprocess_data(df, target_column, feature_columns)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Store preprocessing info for predictions
        self.last_preprocessor = {
            'feature_columns': feature_columns,
            'target_column': target_column,
            'label_encoders': self.label_encoders.copy(),
            'scaler': self.scaler
        }
        
        results = []
        available_models = self.models[metadata['problem_type']]
        
        print(f"\n🚀 Training {len(selected_models)} models...")
        print("=" * 50)
        
        for model_name in selected_models:
            if model_name not in available_models:
                print(f"⚠️  Skipping {model_name} - not available")
                continue
                
            print(f"📊 Training {model_name}...")
            model = available_models[model_name]
            
            try:
                import time
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                training_time = time.time() - start_time
                
                # Calculate score
                if metadata['problem_type'] == 'classification':
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
                    'metric': metric,
                    'training_time': training_time
                })
                
                # Store trained model for predictions
                self.trained_models[model_name] = model
                
                # Store in database
                self.db.store_model_performance(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    features_used=feature_columns,
                    performance_metrics={f'{metric.lower()}_score': score},
                    training_time=training_time,
                    sample_preds=[],
                    llm_rank=1,
                    is_best=False
                )
                
                print(f"✅ {model_name}: {score:.4f} ({training_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
        
        return results
    
    def show_results_and_analysis(self, results, enhanced_rec, prediction, strategy):
        """Show results and offer prediction testing"""
        
        if not results:
            print("❌ No models completed successfully")
            return
        
        best_result = max(results, key=lambda x: x['score'])
        
        print(f"\n🏆 TRAINING RESULTS:")
        print(f"   Strategy: {strategy}")
        print(f"   🥇 Best model: {best_result['model']}")
        print(f"   📈 Score: {best_result['score']:.4f}")
        
        if len(results) > 1:
            print(f"\n📊 All results:")
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            for i, result in enumerate(sorted_results, 1):
                print(f"   {i}. {result['model']}: {result['score']:.4f}")
        
        # Show AI accuracy
        agent_predicted = enhanced_rec['final_recommendation']['top_choice']
        if agent_predicted == best_result['model']:
            print(f"\n🎯 AI ACCURACY: ✅ Agent was right!")
        else:
            print(f"\n🎯 AI ACCURACY: ⚠️ Agent predicted {agent_predicted}, but {best_result['model']} won")
        
        print(f"\n🎉 Training complete!")
        
        # Offer prediction testing
        self.offer_prediction_testing()
    
    def offer_prediction_testing(self):
        """Let user test predictions on new data"""
        if not self.trained_models:
            print("❌ No trained models available for predictions")
            return
        
        print(f"\n🔮 PREDICTION TESTING")
        print(f"Want to test predictions on new data?")
        print(f"   Y. Yes, let me test some predictions")
        print(f"   N. No, I'm done")
        
        choice = input(f"\nTest predictions? (Y/N): ").strip().upper()
        
        if choice == 'Y':
            self.interactive_prediction_testing()
    
    def interactive_prediction_testing(self):
        """Interactive prediction testing interface"""
        
        print(f"\n🧪 INTERACTIVE PREDICTION TESTING")
        print("=" * 50)
        
        # Show available models
        model_names = list(self.trained_models.keys())
        print(f"📋 Available trained models:")
        for i, model in enumerate(model_names, 1):
            print(f"   {i}. {model}")
        
        # Model selection
        while True:
            try:
                model_idx = int(input(f"\nWhich model to use for predictions? (1-{len(model_names)}): ")) - 1
                if 0 <= model_idx < len(model_names):
                    selected_model = model_names[model_idx]
                    print(f"✅ Using {selected_model} for predictions")
                    break
                else:
                    print(f"❌ Please enter 1-{len(model_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # Show required features
        feature_columns = self.last_preprocessor['feature_columns']
        print(f"\n📝 Required features for prediction:")
        for i, feature in enumerate(feature_columns, 1):
            print(f"   {i}. {feature}")
        
        # Interactive prediction loop
        while True:
            print(f"\n🔍 Enter values for prediction:")
            print(f"   (Type 'done' when finished, 'help' for feature info)")
            
            user_data = {}
            for feature in feature_columns:
                while True:
                    value_input = input(f"   {feature}: ").strip()
                    
                    if value_input.lower() == 'done':
                        if len(user_data) == len(feature_columns):
                            break
                        else:
                            print(f"❌ Please provide all {len(feature_columns)} features")
                            continue
                    
                    if value_input.lower() == 'help':
                        print(f"      💡 {feature} - provide a numeric value")
                        continue
                    
                    try:
                        # Try to convert to number
                        user_data[feature] = float(value_input)
                        break
                    except ValueError:
                        # Keep as string (for categorical)
                        user_data[feature] = value_input
                        break
                
                if value_input.lower() == 'done':
                    break
            
            if len(user_data) == len(feature_columns):
                # Make prediction
                try:
                    prediction_result = self.make_single_prediction(user_data, selected_model)
                    print(f"\n🎯 PREDICTION RESULT:")
                    print(f"   Model: {selected_model}")
                    print(f"   Input: {user_data}")
                    print(f"   Predicted {self.last_preprocessor['target_column']}: {prediction_result}")
                    
                except Exception as e:
                    print(f"❌ Prediction failed: {e}")
            
            # Continue or exit
            continue_choice = input(f"\nMake another prediction? (Y/N): ").strip().upper()
            if continue_choice != 'Y':
                break
        
        print(f"✅ Prediction testing complete!")
    
    def make_single_prediction(self, user_data, model_name):
        """Make a single prediction with user data"""
        
        if model_name not in self.trained_models:
            raise Exception(f"Model {model_name} not found")
        
        # Create DataFrame with user data
        df_pred = pd.DataFrame([user_data])
        
        # Apply same preprocessing as training
        feature_columns = self.last_preprocessor['feature_columns']
        X = df_pred[feature_columns].copy()
        
        # Handle categorical features
        for col in X.columns:
            if col in self.last_preprocessor['label_encoders']:
                encoder = self.last_preprocessor['label_encoders'][col]
                try:
                    X[col] = encoder.transform([X[col].iloc[0]])[0]
                except ValueError:
                    # New category, assign -1
                    X[col] = -1
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.last_preprocessor['scaler'].transform(X)
        
        # Make prediction
        model = self.trained_models[model_name]
        prediction = model.predict(X_scaled)[0]
        
        return prediction
    
    def run_full_analysis(self):
        """Main interface - completely user controlled"""
        print("🤖 USER-CONTROLLED DATA SCIENCE AGENT")
        print("=" * 50)
        print("You have FULL control over everything!")
        
        # 1. Dataset selection
        datasets = self.list_available_datasets()
        if not datasets:
            print("❌ No CSV files found in '../data' folder")
            return
        
        print(f"\n📁 Available datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"   {i}. {dataset}")
        
        while True:
            try:
                dataset_idx = int(input(f"\nSelect dataset (1-{len(datasets)}): ")) - 1
                if 0 <= dataset_idx < len(datasets):
                    selected_dataset = datasets[dataset_idx]
                    break
                else:
                    print(f"❌ Please enter 1-{len(datasets)}")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # 2. Load and preview
        print(f"\n{'='*50}")
        df = self.load_and_preview_dataset(selected_dataset)
        if df is None:
            return
        
        # 3. User selects target (NO AUTO-DETECTION)
        target_column = self.get_user_target_selection(df)
        
        # 4. User selects features (MANUAL)
        feature_columns = self.get_user_feature_selection(df, target_column)
        
        # 5. Get AI recommendations (but user decides)
        metadata = self.extract_metadata(df, target_column)
        print(f"\n🔍 Problem detected: {metadata['problem_type']}")
        
        enhanced_rec = self.get_llm_recommendation(metadata)
        
        # 6. User chooses models
        selected_models, strategy = self.get_user_model_choice(metadata, enhanced_rec)
        if selected_models is None:
            return
        
        print(f"\n🎯 YOUR ANALYSIS SETUP:")
        print(f"   📊 Dataset: {selected_dataset}")
        print(f"   🎯 Target: {target_column}")
        print(f"   🔧 Features: {len(feature_columns)} selected")
        print(f"   🤖 Models: {len(selected_models)} selected")
        print(f"   📋 Strategy: {strategy}")
        
        # 7. Train selected models
        results = self.train_selected_models(df, target_column, feature_columns, selected_models, selected_dataset, metadata)
        
        # 8. Show results and offer prediction testing
        prediction = self.db.predict_performance_range(metadata)
        self.show_results_and_analysis(results, enhanced_rec, prediction, strategy)

def main():
    """Main entry point"""
    try:
        agent = UserControlledMLAgent()
        agent.run_full_analysis()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()