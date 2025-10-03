"""
🔍 LEARNING DATABASE VIEWER
Explore what the agent has learned from past sessions
"""

import sys
from pathlib import Path
from learning_database import LearningDatabase
import json

def view_stats():
    """Show overall statistics"""
    db = LearningDatabase()
    stats = db.get_database_stats()
    
    print("\n" + "="*60)
    print("🧠 LEARNING DATABASE STATISTICS")
    print("="*60 + "\n")
    
    print(f"📊 Total Sessions: {stats['total_sessions']}")
    print(f"💡 Total Recommendations: {stats['total_recommendations']}")
    print(f"   🤖 From LLM: {stats['llm_recommendations']}")
    print(f"   💾 From Database: {stats['database_recommendations']}")
    print(f"📚 Learned Patterns: {stats['learned_patterns']}")
    print(f"👤 User Preferences: {stats['user_preferences']}")
    
    if stats['total_recommendations'] > 0:
        llm_pct = (stats['llm_recommendations'] / stats['total_recommendations']) * 100
        db_pct = (stats['database_recommendations'] / stats['total_recommendations']) * 100
        print(f"\n📈 Recommendation Sources:")
        print(f"   LLM: {llm_pct:.1f}%")
        print(f"   Database: {db_pct:.1f}%")
    
    db.close()

def view_recent_sessions(limit=5):
    """Show recent sessions"""
    db = LearningDatabase()
    cursor = db.conn.cursor()
    
    cursor.execute("""
        SELECT session_id, timestamp, dataset_name, 
               num_rows, num_columns, selected_columns
        FROM visualization_sessions
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    
    sessions = cursor.fetchall()
    
    print("\n" + "="*60)
    print(f"📅 RECENT {limit} SESSIONS")
    print("="*60 + "\n")
    
    for i, session in enumerate(sessions, 1):
        session_id, timestamp, dataset_name, num_rows, num_cols, selected_cols = session
        cols = json.loads(selected_cols)
        
        print(f"{i}. Session: {session_id}")
        print(f"   📅 Time: {timestamp}")
        print(f"   📊 Dataset: {dataset_name} ({num_rows} rows × {num_cols} cols)")
        print(f"   🎯 Columns: {cols[:3]}..." if len(cols) > 3 else f"   🎯 Columns: {cols}")
        
        # Get recommendations for this session
        cursor.execute("""
            SELECT COUNT(*) FROM visualization_recommendations
            WHERE session_id = ?
        """, (session_id,))
        rec_count = cursor.fetchone()[0]
        print(f"   💡 Recommendations: {rec_count}")
        print()
    
    db.close()

def view_learned_patterns():
    """Show learned patterns"""
    db = LearningDatabase()
    cursor = db.conn.cursor()
    
    cursor.execute("""
        SELECT pattern_name, pattern_description, times_seen, 
               last_seen, typical_viz_types, example_datasets
        FROM dataset_patterns
        ORDER BY times_seen DESC
    """)
    
    patterns = cursor.fetchall()
    
    print("\n" + "="*60)
    print("📚 LEARNED PATTERNS")
    print("="*60 + "\n")
    
    if not patterns:
        print("🆕 No patterns learned yet. Keep using the system!")
    else:
        for i, pattern in enumerate(patterns, 1):
            name, desc, times_seen, last_seen, viz_types, examples = pattern
            viz_list = json.loads(viz_types)
            example_list = json.loads(examples)
            
            print(f"{i}. Pattern: {name}")
            print(f"   📝 Description: {desc}")
            print(f"   🔢 Seen: {times_seen} times")
            print(f"   📅 Last seen: {last_seen}")
            print(f"   📊 Typical visualizations: {', '.join(viz_list[:3])}")
            print(f"   📂 Example datasets: {', '.join(example_list[:2])}")
            print()
    
    db.close()

def view_recommendations_for_column(column_name):
    """Show all recommendations made for a specific column name"""
    db = LearningDatabase()
    cursor = db.conn.cursor()
    
    cursor.execute("""
        SELECT r.column_name, r.column_type, r.recommended_viz_type,
               r.recommendation_reasoning, r.was_from_llm, r.confidence_score,
               s.dataset_name, s.timestamp
        FROM visualization_recommendations r
        JOIN visualization_sessions s ON r.session_id = s.session_id
        WHERE r.column_name LIKE ?
        ORDER BY s.timestamp DESC
        LIMIT 10
    """, (f"%{column_name}%",))
    
    recommendations = cursor.fetchall()
    
    print("\n" + "="*60)
    print(f"💡 RECOMMENDATIONS FOR '{column_name}'")
    print("="*60 + "\n")
    
    if not recommendations:
        print(f"❌ No recommendations found for column containing '{column_name}'")
    else:
        for i, rec in enumerate(recommendations, 1):
            col, col_type, viz_type, reasoning, from_llm, confidence, dataset, timestamp = rec
            
            print(f"{i}. Column: {col} ({col_type})")
            print(f"   📊 Dataset: {dataset}")
            print(f"   📅 Time: {timestamp}")
            print(f"   🎨 Recommended: {viz_type}")
            print(f"   {'🤖 Source: LLM' if from_llm else '💾 Source: Database'}")
            if confidence:
                print(f"   ✅ Confidence: {confidence*100:.0f}%")
            if reasoning:
                print(f"   💭 Reasoning: {reasoning[:100]}...")
            print()
    
    db.close()

def main():
    print("\n" + "="*60)
    print("🔍 LEARNING DATABASE VIEWER")
    print("="*60)
    
    while True:
        print("\nWhat would you like to explore?")
        print("  1. 📊 Overall Statistics")
        print("  2. 📅 Recent Sessions")
        print("  3. 📚 Learned Patterns")
        print("  4. 🔍 Search Recommendations for Column")
        print("  5. 🚪 Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == "1":
            view_stats()
        elif choice == "2":
            try:
                limit = input("How many recent sessions? (default 5): ").strip()
                limit = int(limit) if limit else 5
                view_recent_sessions(limit)
            except:
                view_recent_sessions(5)
        elif choice == "3":
            view_learned_patterns()
        elif choice == "4":
            column_name = input("Enter column name to search: ").strip()
            if column_name:
                view_recommendations_for_column(column_name)
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Try again!")

if __name__ == "__main__":
    main()
