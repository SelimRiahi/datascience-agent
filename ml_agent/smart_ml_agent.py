import pandas as pd
import numpy as np
import json
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

# Import high-performance models
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# Import ML Database for intelligence
from ml_database import MLDatabase

# Import OpenAI and secrets (same as data cleaning agent)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_quality'))
from openai import OpenAI
from secrets_config import AZURE_API_KEY, AZURE_ENDPOINT, MODEL_NAME

import warnings
import time
warnings.filterwarnings('ignore')

class SmartMLAgent:
    def __init__(self):
        # Initialize ML Intelligence Database
        self.db = MLDatabase()
        print("🧠 ML Intelligence Database connected!")
        
        # Initialize Azure OpenAI client (same as data cleaning agent)
        if AZURE_ENDPOINT.endswith("/openai/v1/"):
            base_url = AZURE_ENDPOINT
        elif "models" in AZURE_ENDPOINT:
            base_url = AZURE_ENDPOINT.replace("/models", "/openai/v1/")
        else:
            base_url = f"{AZURE_ENDPOINT}/openai/v1/"
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=AZURE_API_KEY
        )
        self.model_name = MODEL_NAME
        
        # Test API connectivity
        print("🔗 Testing API connectivity...")
        try:
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.1
            )
            print("✅ API connection successful!")
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            raise Exception("Cannot proceed without OpenAI API access")
        self.models = {
            'classification': {
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
                'SVC': SVC(random_state=42),
                'XGBClassifier': XGBClassifier(random_state=42, eval_metric='logloss'),
                'LGBMClassifier': LGBMClassifier(random_state=42, verbose=-1)
            },
            'regression': {
                'LinearRegression': LinearRegression(),
                'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
                'SVR': SVR(),
                'XGBRegressor': XGBRegressor(random_state=42),
                'LGBMRegressor': LGBMRegressor(random_state=42, verbose=-1)
            }
        }
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def extract_metadata(self, df, target_column):
        """Extract comprehensive dataset metadata"""
        metadata = {
            'n_rows': df.shape[0],
            'n_columns': df.shape[1],
            'target_column': target_column,
            'columns': []
        }
        
        # Analyze each column
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_ratio': float(df[col].isnull().mean()),
                'unique_values': int(df[col].nunique())
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['type'] = 'numeric'
                col_info['stats'] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            else:
                col_info['type'] = 'categorical'
                # Convert values to string to avoid JSON serialization issues
                top_values = df[col].value_counts().head(5)
                col_info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
            
            metadata['columns'].append(col_info)
        
        # Determine problem type
        target_unique = df[target_column].nunique()
        if pd.api.types.is_numeric_dtype(df[target_column]) and target_unique > 20:
            metadata['problem_type'] = 'regression'
        else:
            metadata['problem_type'] = 'classification'
            # Convert class distribution to string keys to avoid JSON issues
            class_dist = df[target_column].value_counts().to_dict()
            metadata['class_distribution'] = {str(k): int(v) for k, v in class_dist.items()}
        
        return metadata
    
    def get_llm_recommendation(self, metadata):
        """Use LLM + Database intelligence to recommend best ML models"""
        print("🤖 Getting intelligent model recommendations...")
        
        # Get pure LLM recommendation first
        prompt = f"""
You are an expert data scientist. Based on the following dataset metadata, recommend the TOP 3 best machine learning models to try, in order of priority.

DATASET METADATA:
{json.dumps(metadata, indent=2, default=str)}

Consider:
- Problem type (classification/regression)
- Dataset size
- Number of features
- Missing data
- Class balance (if classification)
- Feature types

Respond with ONLY a JSON array of model names in order of priority:
For classification: ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression", "SVC"]
For regression: ["XGBRegressor", "LGBMRegressor", "RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression", "SVR"]

Response format: ["Model1", "Model2", "Model3"]
"""
        
        print(f"📤 Sending request to {self.model_name}...")
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert data scientist. Respond with only the JSON array of model names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        response = completion.choices[0].message.content.strip()
        print(f"📥 LLM Response: {response}")
        
        try:
            llm_recommendations = json.loads(response)
            print(f"✅ Successfully parsed LLM recommendation: {llm_recommendations}")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response}")
            raise Exception(f"LLM returned invalid JSON: {response}")
        
        # Enhance with database intelligence
        try:
            enhanced_recommendations = self.db.enhance_recommendations_with_history(llm_recommendations, metadata)
            return enhanced_recommendations
        except Exception as e:
            print(f"⚠️ Database enhancement failed: {e}")
            print("🔄 Falling back to pure LLM recommendations")
            # Return in the expected format
            return {
                'confidence_level': 'LOW',
                'final_recommendation': {
                    'top_choice': llm_recommendations[0] if llm_recommendations else 'RandomForestRegressor',
                    'second_choice': llm_recommendations[1] if len(llm_recommendations) > 1 else 'XGBRegressor'
                },
                'analysis': {
                    'reasoning': 'No historical data available - using pure LLM recommendations'
                }
            }
    

    def preprocess_data(self, df, target_column, feature_columns=None):
        """Preprocess the dataset"""
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical features
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if not pd.api.types.is_numeric_dtype(y):
            if 'target' not in self.label_encoders:
                self.label_encoders['target'] = LabelEncoder()
            y = self.label_encoders['target'].fit_transform(y.astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_and_evaluate(self, df, target_column, feature_columns=None, file_path="unknown"):
        """Complete ML pipeline with database storage"""
        print(f"🤖 SMART ML AGENT")
        print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Target: {target_column}")
        print("=" * 50)
        
        # Store dataset in database
        dataset_id = self.db.store_dataset_metadata(df, target_column, file_path)
        
        # Extract metadata
        print("🔍 Step 1: Analyzing dataset metadata...")
        metadata = self.extract_metadata(df, target_column)
        print(f"Problem type: {metadata['problem_type']}")
        
        # Show database stats
        stats = self.db.get_database_stats()
        print(f"📊 Database: {stats['datasets']} datasets, {stats['model_runs']} runs, Best: {stats['best_model']}")
        
        # 🔮 Performance prediction
        performance_prediction = self.db.predict_performance_range(metadata)
        if performance_prediction:
            metric_name = "Accuracy" if metadata['problem_type'] == 'classification' else "R²"
            print(f"🔮 PERFORMANCE PREDICTION:")
            print(f"   Expected {metric_name}: {performance_prediction['expected_avg']:.3f}")
            print(f"   Range: {performance_prediction['range_min']:.3f} - {performance_prediction['range_max']:.3f}")
            print(f"   Confidence: {performance_prediction['confidence']}")
        
        # Get intelligent recommendations
        print("🧠 Step 2: Getting model recommendations...")
        enhanced_rec = self.get_llm_recommendation(metadata)
        
        # 🚀 ACT ON INTELLIGENCE - Make decisive choices based on confidence
        if enhanced_rec['confidence_level'] == 'HIGH':
            # HIGH CONFIDENCE: Test only the top model (saves ~60% time)
            selected_models = [enhanced_rec['final_recommendation']['top_choice']]
            print(f"🎯 HIGH CONFIDENCE MODE: Testing only {enhanced_rec['final_recommendation']['top_choice']}")
            print(f"   Reason: {enhanced_rec['analysis']['reasoning']}")
            print(f"   ⚡ TIME SAVED: ~60% (testing 1/{len(self.models[metadata['problem_type']])} models)")
            
        elif enhanced_rec['confidence_level'] == 'MEDIUM':
            # MEDIUM CONFIDENCE: Test top 2 models (saves ~33% time)
            top_2 = [enhanced_rec['final_recommendation']['top_choice'], 
                    enhanced_rec['final_recommendation']['second_choice']]
            selected_models = top_2
            print(f"🎯 MEDIUM CONFIDENCE MODE: Testing top 2 models {top_2}")
            print(f"   Reason: {enhanced_rec['analysis']['reasoning']}")
            print(f"   ⚡ TIME SAVED: ~33% (testing 2/{len(self.models[metadata['problem_type']])} models)")
            
        else:
            # LOW CONFIDENCE: Test all models but in smart order
            all_models = list(self.models[metadata['problem_type']].keys())
            # Reorder based on intelligence
            final_rec = enhanced_rec['final_recommendation']
            smart_order = []
            if final_rec['top_choice'] in all_models:
                smart_order.append(final_rec['top_choice'])
                all_models.remove(final_rec['top_choice'])
            if final_rec['second_choice'] in all_models:
                smart_order.append(final_rec['second_choice'])
                all_models.remove(final_rec['second_choice'])
            smart_order.extend(all_models)
            selected_models = smart_order
            print(f"🎯 LOW CONFIDENCE MODE: Testing all models in smart order")
            print(f"   Order: {smart_order}")
            print(f"   Reason: {enhanced_rec['analysis']['reasoning']}")
        
        recommended_models = selected_models
        print(f"📋 Final model selection: {recommended_models}")
        
        # Store feature selection
        if feature_columns:
            available_features = [col for col in df.columns if col != target_column]
            self.db.store_feature_selection(
                dataset_id, available_features, feature_columns, 'user'
            )
        
        # Preprocess data
        print("🔧 Step 3: Preprocessing data...")
        X, y = self.preprocess_data(df, target_column, feature_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate models
        print("🚀 Step 4: Training and evaluating models...")
        results = []
        
        problem_type = metadata['problem_type']
        available_models = self.models[problem_type]
        
        for rank, model_name in enumerate(recommended_models):
            if model_name not in available_models:
                continue
                
            print(f"\n📊 Training {model_name}...")
            model = available_models[model_name]
            
            try:
                # Time the training
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                training_time = time.time() - start_time
                
                # Calculate metrics
                if problem_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                    metric = "Accuracy"
                    performance_metrics = {'accuracy_score': score}
                else:
                    score = r2_score(y_test, y_pred)
                    metric = "R² Score"
                    performance_metrics = {'r2_score': score}
                
                results.append({
                    'model': model_name,
                    'score': score,
                    'metric': metric
                })
                
                # Sample predictions for storage
                sample_preds = []
                y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
                for i in range(min(5, len(y_test_array))):
                    sample_preds.append({
                        'actual': float(y_test_array[i]), 
                        'predicted': float(y_pred[i])
                    })
                
                # Store model performance in database
                self.db.store_model_performance(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    features_used=feature_columns or [col for col in df.columns if col != target_column],
                    performance_metrics=performance_metrics,
                    training_time=training_time,
                    sample_preds=sample_preds,
                    llm_rank=rank + 1,
                    is_best=False  # Will update later
                )
                
                print(f"✅ {model_name} - {metric}: {score:.4f}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
        
        # Find best model and update database
        if results:
            best_result = max(results, key=lambda x: x['score'])
            print(f"\n🏆 BEST MODEL: {best_result['model']}")
            print(f"📈 {best_result['metric']}: {best_result['score']:.4f}")
            
            # Update best model flag in database
            conn = self.db.db.connect(self.db.db_path) if hasattr(self.db, 'db') else sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE model_runs 
                SET is_best_model = 1 
                WHERE dataset_id = ? AND model_name = ?
            ''', (dataset_id, best_result['model']))
            conn.commit()
            conn.close()
            
            # Store LLM recommendation performance
            self.db.store_llm_recommendation(
                dataset_id=dataset_id,
                metadata_sent=metadata,
                recommended_models=[enhanced_rec['final_recommendation']['top_choice'], 
                                  enhanced_rec['final_recommendation']['second_choice']],
                actual_best=best_result['model']
            )
            
            # Show sample predictions
            best_model = available_models[best_result['model']]
            sample_predictions = best_model.predict(X_test[:5])
            print(f"\n📋 Sample Predictions:")
            y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
            for i in range(5):
                print(f"   Actual: {y_test_array[i]}, Predicted: {sample_predictions[i]}")
        
        return results
    
    def _make_predictions(self, df, model_name):
        """Make predictions on new data using the specified model"""
        # Preprocess the data (same way as training)
        X_processed = df.copy()
        
        # Handle categorical features using existing encoders
        for col in X_processed.columns:
            if not pd.api.types.is_numeric_dtype(X_processed[col]):
                if col in self.label_encoders:
                    # Use existing encoder
                    try:
                        X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
                    except ValueError:
                        # Handle unseen categories by encoding as -1
                        X_processed[col] = -1
                else:
                    # New categorical column, encode as -1
                    X_processed[col] = -1
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Scale features using existing scaler
        X_scaled = self.scaler.transform(X_processed)
        
        # Get the model and make predictions
        problem_type = 'classification' if 'Classifier' in model_name else 'regression'
        model = self.models[problem_type][model_name]
        predictions = model.predict(X_scaled)
        
        return predictions

# Example usage - removed hardcoded data
if __name__ == "__main__":
    print("🤖 Smart ML Agent ready!")
    print("Use DatasetLoader to load your data and run the agent.")