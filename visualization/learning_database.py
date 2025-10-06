"""
🧠 VISUALIZATION LEARNING DATABASE
Stores patterns, preferences, and effectiveness to reduce LLM calls over time

Database Structure:
1. visualization_sessions: Every dashboard session
2. visualization_recommendations: What was recommended and why
3. dataset_patterns: Recognized dataset patterns
4. user_preferences: Learned user behavior
5. visualization_effectiveness: What worked well
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd

class LearningDatabase:
    """Database to store and learn from visualization sessions"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            # Use same location as ML agent database
            db_path = Path(__file__).parent.parent / "ml_agent" / "visualization_learning.db"
        
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """Create database and tables if they don't exist"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Table 1: Visualization Sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visualization_sessions (
                session_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_fingerprint TEXT NOT NULL,
                num_rows INTEGER,
                num_columns INTEGER,
                selected_columns TEXT NOT NULL,
                column_types TEXT NOT NULL,
                session_metadata TEXT
            )
        """)
        
        # Table 2: Visualization Recommendations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visualization_recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                column_name TEXT NOT NULL,
                column_type TEXT NOT NULL,
                recommended_viz_type TEXT NOT NULL,
                recommendation_reasoning TEXT,
                was_from_llm BOOLEAN NOT NULL,
                was_from_database BOOLEAN NOT NULL,
                confidence_score REAL,
                effectiveness_score REAL,
                user_interaction_time REAL,
                was_useful BOOLEAN,
                FOREIGN KEY (session_id) REFERENCES visualization_sessions(session_id)
            )
        """)
        
        # Table 3: Dataset Patterns (learned over time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE NOT NULL,
                pattern_description TEXT,
                column_signature TEXT NOT NULL,
                typical_viz_types TEXT NOT NULL,
                success_rate REAL DEFAULT 1.0,
                times_seen INTEGER DEFAULT 1,
                last_seen TEXT,
                example_datasets TEXT
            )
        """)
        
        # Table 4: User Preferences (personalization)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_used TEXT,
                UNIQUE(preference_type, preference_key)
            )
        """)
        
        # Table 5: Visualization Effectiveness (what works well)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visualization_effectiveness (
                effectiveness_id INTEGER PRIMARY KEY AUTOINCREMENT,
                column_type TEXT NOT NULL,
                data_characteristics TEXT NOT NULL,
                viz_type TEXT NOT NULL,
                avg_effectiveness REAL NOT NULL,
                times_used INTEGER DEFAULT 1,
                last_updated TEXT
            )
        """)
        
        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset_fingerprint 
            ON visualization_sessions(dataset_fingerprint)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_column_name 
            ON visualization_recommendations(column_name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_signature 
            ON dataset_patterns(column_signature)
        """)
        
        self.conn.commit()
        print(f"✅ Learning database initialized at: {self.db_path}")
    
    def create_dataset_fingerprint(self, df, selected_columns):
        """
        Create a unique fingerprint for this dataset based on:
        - Column names and types
        - Data distribution characteristics
        - Statistical properties
        """
        fingerprint_data = {
            'columns': selected_columns,
            'types': {col: str(df[col].dtype) for col in selected_columns},
            'shapes': {
                col: {
                    'nunique': int(df[col].nunique()),
                    'null_pct': float(df[col].isnull().sum() / len(df) * 100),
                    'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                    'is_datetime': pd.api.types.is_datetime64_any_dtype(df[col])
                }
                for col in selected_columns
            }
        }
        
        # Create hash of this structure
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def record_session(self, dataset_name, df, selected_columns, group_structure=None):
        """
        Record a new visualization session
        Returns: session_id
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        fingerprint = self.create_dataset_fingerprint(df, selected_columns)
        
        column_types = {col: str(df[col].dtype) for col in selected_columns}
        
        metadata = {
            'group_structure': group_structure,
            'timestamp': datetime.now().isoformat()
        }
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO visualization_sessions 
            (session_id, timestamp, dataset_name, dataset_fingerprint, 
             num_rows, num_columns, selected_columns, column_types, session_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            datetime.now().isoformat(),
            dataset_name,
            fingerprint,
            len(df),
            len(selected_columns),
            json.dumps(selected_columns),
            json.dumps(column_types),
            json.dumps(metadata)
        ))
        
        self.conn.commit()
        print(f"📝 Recorded session: {session_id}")
        return session_id
    
    def record_recommendation(self, session_id, column_name, column_type, 
                            viz_type, reasoning, from_llm=True, 
                            confidence=None):
        """
        Record a visualization recommendation
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO visualization_recommendations
            (session_id, column_name, column_type, recommended_viz_type,
             recommendation_reasoning, was_from_llm, was_from_database,
             confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            column_name,
            column_type,
            viz_type,
            reasoning,
            from_llm,
            not from_llm,
            confidence
        ))
        
        self.conn.commit()
    
    def find_similar_sessions(self, fingerprint, limit=5):
        """
        Find previous sessions with similar datasets
        Returns list of (session_id, dataset_name, similarity_score)
        """
        cursor = self.conn.cursor()
        
        # Exact match first
        cursor.execute("""
            SELECT session_id, dataset_name, selected_columns, column_types
            FROM visualization_sessions
            WHERE dataset_fingerprint = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (fingerprint, limit))
        
        results = cursor.fetchall()
        
        if results:
            return [
                {
                    'session_id': r[0],
                    'dataset_name': r[1],
                    'selected_columns': json.loads(r[2]),
                    'column_types': json.loads(r[3]),
                    'similarity': 1.0  # Exact match
                }
                for r in results
            ]
        
        return []
    
    def get_recommendations_for_session(self, session_id):
        """
        Get all recommendations made in a session
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT column_name, column_type, recommended_viz_type,
                   recommendation_reasoning, was_from_llm, confidence_score
            FROM visualization_recommendations
            WHERE session_id = ?
        """, (session_id,))
        
        results = cursor.fetchall()
        return [
            {
                'column_name': r[0],
                'column_type': r[1],
                'viz_type': r[2],
                'reasoning': r[3],
                'from_llm': bool(r[4]),
                'confidence': r[5]
            }
            for r in results
        ]
    
    def learn_pattern(self, pattern_name, column_signature, viz_types, 
                     dataset_example):
        """
        Store or update a learned pattern
        """
        cursor = self.conn.cursor()
        
        # Check if pattern exists
        cursor.execute("""
            SELECT pattern_id, times_seen, example_datasets
            FROM dataset_patterns
            WHERE pattern_name = ?
        """, (pattern_name,))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing pattern
            pattern_id, times_seen, examples = result
            examples_list = json.loads(examples)
            if dataset_example not in examples_list:
                examples_list.append(dataset_example)
            
            cursor.execute("""
                UPDATE dataset_patterns
                SET times_seen = ?,
                    last_seen = ?,
                    example_datasets = ?
                WHERE pattern_id = ?
            """, (
                times_seen + 1,
                datetime.now().isoformat(),
                json.dumps(examples_list[:10]),  # Keep only 10 examples
                pattern_id
            ))
        else:
            # Create new pattern
            cursor.execute("""
                INSERT INTO dataset_patterns
                (pattern_name, pattern_description, column_signature,
                 typical_viz_types, last_seen, example_datasets)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pattern_name,
                f"Auto-detected pattern from {dataset_example}",
                json.dumps(column_signature),
                json.dumps(viz_types),
                datetime.now().isoformat(),
                json.dumps([dataset_example])
            ))
        
        self.conn.commit()
    
    def get_database_stats(self):
        """
        Get statistics about what the database has learned
        """
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM visualization_sessions")
        stats['total_sessions'] = cursor.fetchone()[0]
        
        # Total recommendations
        cursor.execute("SELECT COUNT(*) FROM visualization_recommendations")
        stats['total_recommendations'] = cursor.fetchone()[0]
        
        # LLM vs Database recommendations
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN was_from_llm THEN 1 ELSE 0 END) as llm_count,
                SUM(CASE WHEN was_from_database THEN 1 ELSE 0 END) as db_count
            FROM visualization_recommendations
        """)
        llm, db = cursor.fetchone()
        stats['llm_recommendations'] = llm or 0
        stats['database_recommendations'] = db or 0
        
        # Learned patterns
        cursor.execute("SELECT COUNT(*) FROM dataset_patterns")
        stats['learned_patterns'] = cursor.fetchone()[0]
        
        # User preferences
        cursor.execute("SELECT COUNT(*) FROM user_preferences")
        stats['user_preferences'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    # Test the database
    print("🧪 Testing Learning Database...")
    
    db = LearningDatabase()
    
    # Show stats
    stats = db.get_database_stats()
    print("\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    db.close()
    print("\n✅ Test complete!")
