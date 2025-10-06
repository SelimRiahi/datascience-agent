import sqlite3
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

class MLDatabase:
    """
    Intelligent ML Database for storing and learning from ML experiments
    Tracks datasets, model performance, and provides smart recommendations
    """
    
    def __init__(self, db_path="ml_intelligence.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Datasets table - stores dataset metadata and fingerprints
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_hash TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            file_path TEXT,
            shape_rows INTEGER,
            shape_cols INTEGER,
            target_column TEXT,
            problem_type TEXT,
            feature_types TEXT,
            missing_data_pct REAL,
            unique_target_values INTEGER,
            target_range TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Model runs table - stores individual model performance
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            model_name TEXT NOT NULL,
            features_used TEXT,
            feature_count INTEGER,
            r2_score REAL,
            accuracy_score REAL,
            training_time REAL,
            hyperparameters TEXT,
            sample_predictions TEXT,
            llm_recommendation_rank INTEGER,
            is_best_model BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        ''')
        
        # LLM recommendations table - tracks LLM suggestion accuracy
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            metadata_sent TEXT,
            models_recommended TEXT,
            actual_best_model TEXT,
            recommendation_hit_rate REAL,
            llm_confidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        ''')
        
        # Feature selections table - learns from user feature choices
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_selections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            available_features TEXT,
            selected_features TEXT,
            selection_type TEXT, -- 'user', 'auto', 'all'
            performance_impact REAL,
            feature_importance TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        ''')
        
        # Model intelligence table - learns patterns across datasets
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_pattern_hash TEXT,
            problem_type TEXT,
            dataset_size_range TEXT,
            feature_count_range TEXT,
            best_model TEXT,
            avg_performance REAL,
            confidence_score REAL,
            sample_count INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print(f"🗄️ ML Database initialized: {self.db_path}")
    
    def generate_dataset_hash(self, df, target_column, file_path):
        """Generate unique fingerprint for dataset"""
        # Create hash based on shape, columns, target, and sample data
        hash_input = f"{df.shape}_{list(df.columns)}_{target_column}_{df.head(3).to_string()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def store_dataset_metadata(self, df, target_column, file_path):
        """Store dataset metadata and return dataset ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate dataset fingerprint
        dataset_hash = self.generate_dataset_hash(df, target_column, file_path)
        
        # Calculate metadata
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        unique_target = df[target_column].nunique()
        
        # Determine problem type
        problem_type = 'regression' if (pd.api.types.is_numeric_dtype(df[target_column]) and unique_target > 20) else 'classification'
        
        # Feature types
        feature_types = {}
        for col in df.columns:
            if col != target_column:
                feature_types[col] = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        
        # Target range
        if problem_type == 'regression':
            target_range = f"{df[target_column].min()}-{df[target_column].max()}"
        else:
            top_classes = df[target_column].value_counts().head(3).index.tolist()
            target_range = str(top_classes)
        
        # Check if dataset already exists
        cursor.execute('SELECT id FROM datasets WHERE dataset_hash = ?', (dataset_hash,))
        existing = cursor.fetchone()
        
        if existing:
            dataset_id = existing[0]
            print(f"📋 Dataset already known (ID: {dataset_id})")
        else:
            # Insert new dataset
            cursor.execute('''
            INSERT INTO datasets (
                dataset_hash, name, file_path, shape_rows, shape_cols,
                target_column, problem_type, feature_types, missing_data_pct,
                unique_target_values, target_range
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_hash, Path(file_path).name, file_path,
                df.shape[0], df.shape[1], target_column, problem_type,
                json.dumps(feature_types), missing_pct, unique_target, target_range
            ))
            dataset_id = cursor.lastrowid
            print(f"📋 New dataset registered (ID: {dataset_id})")
        
        conn.commit()
        conn.close()
        return dataset_id
    
    def store_model_performance(self, dataset_id, model_name, features_used, performance_metrics, 
                              training_time=0, hyperparams=None, sample_preds=None, llm_rank=None, is_best=False):
        """Store individual model performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract performance scores
        r2_score = performance_metrics.get('r2_score')
        accuracy = performance_metrics.get('accuracy_score')
        
        cursor.execute('''
        INSERT INTO model_runs (
            dataset_id, model_name, features_used, feature_count,
            r2_score, accuracy_score, training_time, hyperparameters,
            sample_predictions, llm_recommendation_rank, is_best_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id, model_name, json.dumps(features_used), len(features_used),
            r2_score, accuracy, training_time, json.dumps(hyperparams) if hyperparams else None,
            json.dumps(sample_preds) if sample_preds else None, llm_rank, is_best
        ))
        
        conn.commit()
        conn.close()
        print(f"🤖 Stored performance: {model_name} - Score: {r2_score or accuracy}")
    
    def store_llm_recommendation(self, dataset_id, metadata_sent, recommended_models, actual_best):
        """Store LLM recommendation for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate hit rate
        hit_rate = 1.0 if actual_best in recommended_models else 0.0
        if actual_best in recommended_models:
            hit_rate += (3 - recommended_models.index(actual_best)) / 3  # Bonus for rank
        
        cursor.execute('''
        INSERT INTO llm_recommendations (
            dataset_id, metadata_sent, models_recommended, 
            actual_best_model, recommendation_hit_rate
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            dataset_id, json.dumps(metadata_sent), json.dumps(recommended_models),
            actual_best, hit_rate
        ))
        
        conn.commit()
        conn.close()
        print(f"🧠 LLM recommendation logged - Hit rate: {hit_rate:.2f}")
    
    def store_feature_selection(self, dataset_id, available_features, selected_features, selection_type, performance_impact=None):
        """Store user feature selection patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO feature_selections (
            dataset_id, available_features, selected_features,
            selection_type, performance_impact
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            dataset_id, json.dumps(available_features), json.dumps(selected_features),
            selection_type, performance_impact
        ))
        
        conn.commit()
        conn.close()
        print(f"🔧 Feature selection logged: {len(selected_features)}/{len(available_features)} features")
    
    def get_similar_datasets(self, current_metadata, limit=5):
        """Find similar datasets for intelligent recommendations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Look for datasets with similar characteristics
        query = '''
        SELECT d.*, AVG(mr.r2_score) as avg_performance, COUNT(mr.id) as run_count
        FROM datasets d
        LEFT JOIN model_runs mr ON d.id = mr.dataset_id
        WHERE d.problem_type = ? 
        AND d.shape_rows BETWEEN ? AND ?
        AND d.shape_cols BETWEEN ? AND ?
        GROUP BY d.id
        HAVING run_count > 0
        ORDER BY avg_performance DESC
        LIMIT ?
        '''
        
        # Define similarity ranges (±50% for rows, ±3 for columns)
        row_min = max(1, int(current_metadata['n_rows'] * 0.5))
        row_max = int(current_metadata['n_rows'] * 1.5)
        col_min = max(1, current_metadata['n_columns'] - 3)
        col_max = current_metadata['n_columns'] + 3
        
        cursor.execute(query, (
            current_metadata['problem_type'], row_min, row_max,
            col_min, col_max, limit
        ))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            print(f"🔍 Found {len(results)} similar datasets")
        return results
    
    def get_best_models_for_pattern(self, problem_type, dataset_size, feature_count, limit=3):
        """Get historically best models for this type of problem"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Size categories
        size_category = 'small' if dataset_size < 1000 else ('medium' if dataset_size < 10000 else 'large')
        
        query = '''
        SELECT mr.model_name, AVG(mr.r2_score) as avg_score, COUNT(*) as usage_count
        FROM model_runs mr
        JOIN datasets d ON mr.dataset_id = d.id
        WHERE d.problem_type = ?
        AND ((? = 'small' AND d.shape_rows < 1000) OR 
             (? = 'medium' AND d.shape_rows BETWEEN 1000 AND 9999) OR 
             (? = 'large' AND d.shape_rows >= 10000))
        GROUP BY mr.model_name
        HAVING usage_count >= 2
        ORDER BY avg_score DESC, usage_count DESC
        LIMIT ?
        '''
        
        cursor.execute(query, (problem_type, size_category, size_category, size_category, limit))
        results = cursor.fetchall()
        conn.close()
        
        return [(row[0], row[1], row[2]) for row in results]
    
    def get_database_stats(self):
        """Get intelligence database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Dataset count
        cursor.execute('SELECT COUNT(*) FROM datasets')
        stats['datasets'] = cursor.fetchone()[0]
        
        # Model runs count
        cursor.execute('SELECT COUNT(*) FROM model_runs')
        stats['model_runs'] = cursor.fetchone()[0]
        
        # Best performing model overall
        cursor.execute('''
        SELECT model_name, AVG(r2_score) as avg_score, COUNT(*) as count
        FROM model_runs 
        WHERE r2_score IS NOT NULL
        GROUP BY model_name
        ORDER BY avg_score DESC
        LIMIT 1
        ''')
        best = cursor.fetchone()
        stats['best_model'] = f"{best[0]} (avg: {best[1]:.3f})" if best else "None"
        
        # LLM accuracy
        cursor.execute('SELECT AVG(recommendation_hit_rate) FROM llm_recommendations')
        llm_acc = cursor.fetchone()[0]
        stats['llm_accuracy'] = f"{llm_acc:.2%}" if llm_acc else "No data"
        
        conn.close()
        return stats
    
    def enhance_recommendations_with_history(self, llm_recommendations, current_metadata):
        """🤖 INTELLIGENT AGENT: Make decisive recommendations based on historical data"""
        historical_best = self.get_best_models_for_pattern(
            current_metadata['problem_type'],
            current_metadata['n_rows'],
            current_metadata['n_columns']
        )
        
        if not historical_best:
            print("🧠 No historical data - using pure LLM recommendations")
            # Return in expected format with LOW confidence
            return {
                'confidence_level': 'LOW',
                'final_recommendation': {
                    'top_choice': llm_recommendations[0] if llm_recommendations else 'RandomForestRegressor',
                    'second_choice': llm_recommendations[1] if len(llm_recommendations) > 1 else 'XGBRegressor'
                },
                'analysis': {
                    'reasoning': 'No historical data available - following LLM suggestions'
                }
            }
        
        # Get historical performance statistics
        best_model, best_score, best_usage = historical_best[0]
        second_model, second_score, second_usage = historical_best[1] if len(historical_best) > 1 else (None, 0, 0)
        
        # Calculate confidence metrics
        total_usage = sum([usage for _, _, usage in historical_best])
        confidence = best_usage / total_usage if total_usage > 0 else 0
        performance_gap = (best_score - second_score) if second_model else 0.1
        
        print(f"🧠 AGENT INTELLIGENCE ANALYSIS:")
        print(f"📊 Historical data: {len(historical_best)} models, {total_usage} total runs")
        print(f"🏆 Best performer: {best_model} ({best_score:.3f}, {best_usage} uses)")
        print(f"📈 Confidence: {confidence:.1%}, Performance gap: {performance_gap:.3f}")
        
        # 🤖 INTELLIGENT DECISION MAKING - Make it more decisive
        if confidence >= 0.4 and performance_gap >= 0.05 and best_usage >= 3:
            # HIGH CONFIDENCE: Skip inferior models, focus on the winner
            print(f"🎯 HIGH CONFIDENCE MODE: {best_model} dominates historically")
            print(f"⚡ DECISION: Testing only {best_model} (saves ~60% training time)")
            return {
                'confidence_level': 'HIGH',
                'final_recommendation': {
                    'top_choice': best_model,
                    'second_choice': second_model if second_model else 'XGBRegressor'
                },
                'analysis': {
                    'reasoning': f'HIGH CONFIDENCE: {best_model} dominates with {confidence:.1%} usage and {performance_gap:.3f} performance gap'
                }
            }
            
        elif confidence >= 0.3 and performance_gap >= 0.03 and best_usage >= 2:
            # MEDIUM CONFIDENCE: Test top 2 performers, skip worst
            recommended = [best_model]
            if second_model and second_model in llm_recommendations:
                recommended.append(second_model)
            print(f"🎯 MEDIUM CONFIDENCE MODE: {best_model} + {second_model} are proven winners")
            print(f"⚡ DECISION: Testing top 2 models (saves ~33% training time)")
            return {
                'confidence_level': 'MEDIUM',
                'final_recommendation': {
                    'top_choice': best_model,
                    'second_choice': second_model if second_model else 'XGBRegressor'
                },
                'analysis': {
                    'reasoning': f'MEDIUM CONFIDENCE: {best_model} + {second_model} are proven winners'
                }
            }
            
        elif confidence >= 0.4 and len(historical_best) >= 2:
            # SMART REORDERING: Prioritize historical winners
            historical_models = [model[0] for model in historical_best[:2]]
            enhanced = []
            
            # Add historical winners first (in order of performance)
            for hist_model in historical_models:
                if hist_model in llm_recommendations:
                    enhanced.append(hist_model)
                    
            # Add remaining LLM models
            for llm_model in llm_recommendations:
                if llm_model not in enhanced:
                    enhanced.append(llm_model)
                    
            print(f"🎯 SMART REORDERING: Prioritizing {historical_models[0]} → {historical_models[1]}")
            print(f"⚡ DECISION: Testing all 3 but prioritizing proven performers")
            return enhanced[:3]
            
        else:
            # LOW CONFIDENCE: Fall back to LLM with historical context
            print(f"🧠 INSUFFICIENT DATA: Using LLM recommendations with historical awareness")
            print(f"📊 Historical hint: {best_model} performed best ({best_score:.3f})")
            # Return in expected format with LOW confidence but historical awareness
            return {
                'confidence_level': 'LOW',
                'final_recommendation': {
                    'top_choice': best_model,  # Put historical best first
                    'second_choice': llm_recommendations[0] if llm_recommendations else 'RandomForestRegressor'
                },
                'analysis': {
                    'reasoning': f'Insufficient data for high confidence. Historical hint: {best_model} performed best ({best_score:.3f})'
                }
            }
    
    def predict_performance_range(self, current_metadata):
        """🔮 Predict expected performance range based on historical data"""
        historical_best = self.get_best_models_for_pattern(
            current_metadata['problem_type'],
            current_metadata['n_rows'],
            current_metadata['n_columns']
        )
        
        if not historical_best or len(historical_best) < 2:
            return None
            
        scores = [score for _, score, _ in historical_best]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        return {
            'expected_avg': avg_score,
            'range_min': min_score,
            'range_max': max_score,
            'confidence': 'High' if len(historical_best) >= 3 else 'Medium'
        }