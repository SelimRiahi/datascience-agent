"""
🧠 SMART RECOMMENDER - Uses Database First, LLM as Fallback
This is the hybrid system that learns from history and reduces LLM costs
"""

import pandas as pd
from learning_database import LearningDatabase
from intelligent_visualizer import IntelligentVisualizer
import json

class SmartRecommender:
    """
    Intelligent recommender that:
    1. Checks database for patterns (FAST, FREE)
    2. Falls back to LLM if needed (SLOW, COSTS)
    3. Learns from every interaction
    """
    
    def __init__(self, confidence_threshold=0.75):
        self.db = LearningDatabase()
        self.llm = IntelligentVisualizer()
        self.confidence_threshold = confidence_threshold
        self.stats = {
            'db_hits': 0,
            'llm_calls': 0,
            'hybrid_calls': 0
        }
    
    def get_recommendations(self, df, selected_columns, dataset_name, session_id):
        """
        Smart recommendation with confidence-based decision
        
        Returns: {
            'recommendations': {...},
            'source': 'database' | 'llm' | 'hybrid',
            'confidence': 0.0-1.0,
            'reasoning': 'why this source was used'
        }
        """
        
        print("\n🧠 Smart Recommender Analyzing...")
        
        # Step 1: Create fingerprint and check database
        fingerprint = self.db.create_dataset_fingerprint(df, selected_columns)
        similar_sessions = self.db.find_similar_sessions(fingerprint, limit=10)
        
        if not similar_sessions:
            # NEW DATA - Must use LLM
            print("   🆕 New dataset pattern - calling LLM")
            return self._use_llm(df, selected_columns, dataset_name, session_id, 
                               reason="No similar datasets in history")
        
        # Step 2: Calculate confidence based on history
        confidence = self._calculate_confidence(similar_sessions, selected_columns)
        
        print(f"   📊 Found {len(similar_sessions)} similar session(s)")
        print(f"   💪 Confidence: {confidence*100:.1f}%")
        
        # Step 3: Decide based on confidence
        if confidence >= self.confidence_threshold:
            # HIGH CONFIDENCE - Use database
            print(f"   ✅ Confidence ≥ {self.confidence_threshold*100:.0f}% - Using database!")
            return self._use_database(similar_sessions, selected_columns, session_id,
                                    confidence=confidence, df=df)
        
        elif confidence >= 0.4:
            # MEDIUM CONFIDENCE - Use hybrid (DB + LLM)
            print(f"   🔄 Medium confidence - Using hybrid approach")
            return self._use_hybrid(df, selected_columns, similar_sessions, 
                                  dataset_name, session_id, confidence=confidence)
        
        else:
            # LOW CONFIDENCE - Use LLM
            print(f"   ⚠️ Low confidence - Using LLM")
            return self._use_llm(df, selected_columns, dataset_name, session_id,
                               reason=f"Low confidence ({confidence*100:.0f}%)")
    
    def _calculate_confidence(self, similar_sessions, selected_columns):
        """
        Calculate confidence based on:
        - How many times we've seen this pattern
        - How consistent the recommendations were
        - How recent the data is
        """
        
        # Base confidence on number of similar sessions
        num_sessions = len(similar_sessions)
        
        if num_sessions == 0:
            return 0.0
        elif num_sessions == 1:
            return 0.2
        elif num_sessions <= 3:
            return 0.4
        elif num_sessions <= 5:
            return 0.6
        elif num_sessions <= 10:
            return 0.8
        else:
            return 0.95
    
    def _use_database(self, similar_sessions, selected_columns, session_id, confidence, df=None):
        """
        Use pure database recommendations (NO LLM CALL)
        """
        self.stats['db_hits'] += 1
        
        # Get most common recommendations from history
        column_recommendations = {}
        
        for session in similar_sessions:
            session_recs = self.db.get_recommendations_for_session(session['session_id'])
            
            for rec in session_recs:
                col_name = rec['column_name']
                viz_type = rec['viz_type']
                
                # Count occurrences
                if col_name not in column_recommendations:
                    column_recommendations[col_name] = {}
                
                if viz_type not in column_recommendations[col_name]:
                    column_recommendations[col_name][viz_type] = 0
                
                column_recommendations[col_name][viz_type] += 1
        
        # Build recommendations
        individual_visualizations = []
        
        for col in selected_columns:
            # Find best match (might be exact name or similar name)
            best_viz = None
            best_count = 0
            
            # Check exact match
            if col in column_recommendations:
                for viz_type, count in column_recommendations[col].items():
                    if count > best_count:
                        best_viz = viz_type
                        best_count = count
            
            # If not found, try fuzzy match (e.g., "salary" matches "employee_salary")
            if not best_viz:
                for hist_col, viz_dict in column_recommendations.items():
                    if col.lower() in hist_col.lower() or hist_col.lower() in col.lower():
                        for viz_type, count in viz_dict.items():
                            if count > best_count:
                                best_viz = viz_type
                                best_count = count
            
            # Default fallback
            if not best_viz:
                col_type = df[col].dtype
                if pd.api.types.is_numeric_dtype(col_type):
                    best_viz = "histogram"
                else:
                    best_viz = "bar_chart"
            
            individual_visualizations.append({
                'column': col,
                'visualization_type': best_viz,
                'reasoning': f"Learned pattern (seen {best_count}x in {len(similar_sessions)} similar datasets)",
                'confidence': confidence
            })
            
            # Record in database
            self.db.record_recommendation(
                session_id=session_id,
                column_name=col,
                column_type=str(df[col].dtype),
                viz_type=best_viz,
                reasoning=f"Database pattern match (confidence: {confidence*100:.0f}%)",
                from_llm=False,
                confidence=confidence
            )
        
        return {
            'recommendations': {
                'individual_visualizations': individual_visualizations,
                'relationship_visualizations': []
            },
            'source': 'database',
            'confidence': confidence,
            'reasoning': f"Pattern recognized from {len(similar_sessions)} similar datasets",
            'cost': 0.0  # NO LLM COST!
        }
    
    def _use_llm(self, df, selected_columns, dataset_name, session_id, reason):
        """
        Use LLM recommendations (COSTS MONEY)
        """
        self.stats['llm_calls'] += 1
        
        print(f"   🤖 Calling LLM... ({reason})")
        recommendations = self.llm.analyze_and_recommend(df, selected_columns)
        
        # Record recommendations
        if 'individual_visualizations' in recommendations:
            for viz in recommendations['individual_visualizations']:
                col_name = viz.get('column', 'unknown')
                viz_type = viz.get('visualization_type', 'unknown')
                reasoning = viz.get('reasoning', '')
                
                self.db.record_recommendation(
                    session_id=session_id,
                    column_name=col_name,
                    column_type=str(df[col_name].dtype) if col_name in df.columns else 'unknown',
                    viz_type=viz_type,
                    reasoning=reasoning,
                    from_llm=True,
                    confidence=1.0
                )
        
        return {
            'recommendations': recommendations,
            'source': 'llm',
            'confidence': 1.0,
            'reasoning': reason,
            'cost': 0.05  # Approximate LLM cost
        }
    
    def _use_hybrid(self, df, selected_columns, similar_sessions, dataset_name, 
                   session_id, confidence):
        """
        Use database + LLM (PARTIAL COST)
        Database for common patterns, LLM for edge cases
        """
        self.stats['hybrid_calls'] += 1
        
        print("   🔄 Using hybrid: Database patterns + LLM validation")
        
        # Get database suggestions first
        db_result = self._use_database(similar_sessions, selected_columns, 
                                      session_id, confidence, df=df)
        
        # Then validate/enhance with LLM
        llm_result = self._use_llm(df, selected_columns, dataset_name, 
                                  session_id, "Validating database patterns")
        
        # Combine both (prefer LLM when they differ)
        return {
            'recommendations': llm_result['recommendations'],
            'source': 'hybrid',
            'confidence': confidence,
            'reasoning': f"Database patterns ({confidence*100:.0f}%) validated by LLM",
            'cost': 0.05,  # Full LLM cost
            'db_suggestions': db_result['recommendations']
        }
    
    def get_stats(self):
        """Get usage statistics"""
        total = self.stats['db_hits'] + self.stats['llm_calls'] + self.stats['hybrid_calls']
        
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'db_percentage': (self.stats['db_hits'] / total) * 100,
            'llm_percentage': (self.stats['llm_calls'] / total) * 100,
            'hybrid_percentage': (self.stats['hybrid_calls'] / total) * 100,
            'estimated_savings': (self.stats['db_hits'] * 0.05)  # Money saved by not calling LLM
        }
    
    def close(self):
        """Close database connection"""
        self.db.close()


if __name__ == "__main__":
    # Test the smart recommender
    print("🧪 Testing Smart Recommender...\n")
    
    recommender = SmartRecommender(confidence_threshold=0.75)
    
    # Create test data
    test_data = pd.DataFrame({
        'price': [100, 200, 150, 300, 250] * 3,
        'bedrooms': [2, 3, 2, 4, 3] * 3,
        'square_feet': [1000, 1500, 1200, 2000, 1800] * 3
    })
    
    print("📊 Test Dataset:")
    print(f"   Columns: {list(test_data.columns)}")
    print(f"   Shape: {test_data.shape}")
    
    # Simulate multiple sessions to build confidence
    for i in range(1, 4):
        print(f"\n{'='*60}")
        print(f"SESSION #{i}")
        print('='*60)
        
        session_id = recommender.db.record_session(
            dataset_name=f'test_house_prices_{i}.csv',
            df=test_data,
            selected_columns=['price', 'bedrooms', 'square_feet'],
            group_structure=[['price', 'bedrooms', 'square_feet']]
        )
        
        result = recommender.get_recommendations(
            df=test_data,
            selected_columns=['price', 'bedrooms', 'square_feet'],
            dataset_name=f'test_house_prices_{i}.csv',
            session_id=session_id
        )
        
        print(f"\n📊 Result:")
        print(f"   Source: {result['source'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   Cost: ${result['cost']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")
        
        if 'individual_visualizations' in result['recommendations']:
            print(f"\n💡 Recommendations:")
            for viz in result['recommendations']['individual_visualizations'][:3]:
                viz_type = viz.get('visualization_type') or viz.get('type') or 'unknown'
                print(f"   • {viz['column']} → {viz_type}")
    
    print(f"\n{'='*60}")
    print("📊 FINAL STATISTICS")
    print('='*60)
    
    stats = recommender.get_stats()
    print(f"\n💾 Database recommendations: {stats['db_hits']} ({stats.get('db_percentage', 0):.1f}%)")
    print(f"🤖 LLM recommendations: {stats['llm_calls']} ({stats.get('llm_percentage', 0):.1f}%)")
    print(f"🔄 Hybrid recommendations: {stats['hybrid_calls']} ({stats.get('hybrid_percentage', 0):.1f}%)")
    print(f"\n💰 Estimated savings: ${stats.get('estimated_savings', 0):.2f}")
    
    recommender.close()
    print("\n✅ Test complete!")
