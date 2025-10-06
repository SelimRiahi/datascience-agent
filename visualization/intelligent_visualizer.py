"""
🎨 INTELLIGENT VISUALIZATION AGENT
User selects columns → LLM recommends best visualizations → Auto-generate charts

Features:
- User controls which columns to visualize
- LLM analyzes data characteristics
- Smart visualization recommendations
- Beautiful Streamlit interface
"""

import pandas as pd
import numpy as np
from openai import OpenAI
import json
import os
import sys
from pathlib import Path

class IntelligentVisualizer:
    """AI-powered visualization recommender"""
    
    def __init__(self):
        self.client = self._init_openai_client()
        self.column_analysis = {}
        self.recommendations = {}
    
    def _init_openai_client(self):
        """Initialize OpenAI client (same approach as ML agent)"""
        try:
            # Add parent directory to path to import secrets
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir / 'data_quality'))
            
            from secrets_config import AZURE_API_KEY, AZURE_ENDPOINT, MODEL_NAME
            
            # Construct base URL properly
            if AZURE_ENDPOINT.endswith("/openai/v1/"):
                base_url = AZURE_ENDPOINT
            elif "models" in AZURE_ENDPOINT:
                base_url = AZURE_ENDPOINT.replace("/models", "/openai/v1/")
            else:
                base_url = f"{AZURE_ENDPOINT}/openai/v1/"
            
            client = OpenAI(
                base_url=base_url,
                api_key=AZURE_API_KEY
            )
            
            self.model_name = MODEL_NAME
            
            # Test API connectivity
            test_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.1
            )
            print("✅ Azure OpenAI connected")
            return client
            
        except Exception as e:
            print(f"⚠️ Azure OpenAI not configured: {e}")
            return None
    
    def analyze_column_characteristics(self, df, column):
        """Analyze a single column's characteristics for visualization"""
        
        col_data = df[column]
        
        analysis = {
            'name': column,
            'dtype': str(col_data.dtype),
            'total_count': int(len(col_data)),
            'non_null_count': int(col_data.count()),
            'null_count': int(col_data.isnull().sum()),
            'null_percentage': float((col_data.isnull().sum() / len(col_data)) * 100),
            'unique_count': int(col_data.nunique()),
            'is_numeric': bool(pd.api.types.is_numeric_dtype(col_data)),
            'is_categorical': bool(pd.api.types.is_object_dtype(col_data) or col_data.nunique() < 20),
        }
        
        # Numeric column statistics
        if analysis['is_numeric']:
            analysis.update({
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'range': float(col_data.max() - col_data.min()),
                'has_outliers': bool(self._detect_outliers(col_data)),
                'distribution_shape': str(self._detect_distribution(col_data))
            })
        
        # Categorical column statistics
        if analysis['is_categorical']:
            value_counts = col_data.value_counts()
            analysis.update({
                'top_categories': {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()},
                'category_count': int(len(value_counts)),
                'is_binary': bool(col_data.nunique() == 2),
                'is_balanced': bool(self._check_balance(value_counts))
            })
        
        return analysis
    
    def _detect_outliers(self, series):
        """Detect if column has outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
        return outliers > 0
    
    def _detect_distribution(self, series):
        """Detect distribution shape"""
        skewness = series.skew()
        if abs(skewness) < 0.5:
            return "normal"
        elif skewness > 0.5:
            return "right-skewed"
        else:
            return "left-skewed"
    
    def _check_balance(self, value_counts):
        """Check if categories are balanced"""
        if len(value_counts) < 2:
            return True
        ratio = value_counts.max() / value_counts.min()
        return ratio < 3  # Consider balanced if largest/smallest < 3x
    
    def analyze_column_relationships(self, df, selected_columns):
        """Analyze relationships between selected columns"""
        
        relationships = []
        
        # Analyze pairs of columns
        for i, col1 in enumerate(selected_columns):
            for col2 in selected_columns[i+1:]:
                
                relationship = {
                    'columns': [col1, col2],
                    'both_numeric': (self.column_analysis[col1]['is_numeric'] and 
                                   self.column_analysis[col2]['is_numeric']),
                    'both_categorical': (self.column_analysis[col1]['is_categorical'] and 
                                       self.column_analysis[col2]['is_categorical']),
                    'mixed': (self.column_analysis[col1]['is_numeric'] != 
                            self.column_analysis[col2]['is_numeric'])
                }
                
                # Calculate correlation for numeric pairs
                if relationship['both_numeric']:
                    correlation = df[col1].corr(df[col2])
                    relationship['correlation'] = float(correlation)
                    relationship['correlation_strength'] = self._categorize_correlation(correlation)
                
                relationships.append(relationship)
        
        return relationships
    
    def _categorize_correlation(self, corr):
        """Categorize correlation strength"""
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            return "strong"
        elif abs_corr > 0.4:
            return "moderate"
        else:
            return "weak"
    
    def get_llm_visualization_recommendations(self, df, selected_columns):
        """Ask LLM for intelligent visualization recommendations"""
        
        if not self.client:
            return self._fallback_recommendations(selected_columns)
        
        # Prepare context for LLM
        context = {
            'selected_columns': selected_columns,
            'column_analyses': {col: self.column_analysis[col] for col in selected_columns},
            'relationships': self.analyze_column_relationships(df, selected_columns) if len(selected_columns) > 1 else []
        }
        
        print(f"\n📤 SENDING TO LLM:")
        print(f"=" * 70)
        print(f"📊 COLUMN ANALYSES:")
        print(json.dumps(context['column_analyses'], indent=2))
        print(f"\n🔗 RELATIONSHIPS:")
        print(json.dumps(context['relationships'], indent=2))
        print(f"=" * 70)
        
        prompt = f"""You are an expert data visualization consultant. Analyze the following data columns and recommend the best visualization types.

**Selected Columns for Visualization:**
{json.dumps(context['column_analyses'], indent=2)}

**Column Relationships:**
{json.dumps(context['relationships'], indent=2) if context['relationships'] else 'Single column selected'}

**Your Task:**
For each column and relationship, recommend:
1. **Visualization Type** (e.g., histogram, bar chart, scatter plot, box plot, heatmap, line chart, pie chart)
2. **Reasoning** (why this visualization is best for this data)
3. **Styling Recommendations** (colors, labels, titles, axis configurations)
4. **Key Insights** (what patterns or insights to look for)

**Important Considerations:**
- Numeric data with normal distribution → histogram or box plot
- Numeric data with outliers → box plot to show outliers
- Categorical data with few categories → bar chart or pie chart
- Categorical data with many categories → horizontal bar chart (top N)
- Two numeric columns with correlation → scatter plot
- Categorical vs Numeric → grouped bar chart or box plot by category
- Time series data → line chart
- Multiple numeric columns → correlation heatmap or pair plot

**Output Format (JSON):**
{{
    "individual_visualizations": [
        {{
            "column": "column_name",
            "viz_type": "histogram|bar_chart|box_plot|pie_chart",
            "reasoning": "why this visualization",
            "styling": {{
                "title": "Chart Title",
                "color": "color_scheme",
                "xlabel": "X axis label",
                "ylabel": "Y axis label"
            }},
            "insights_to_look_for": ["insight 1", "insight 2"]
        }}
    ],
    "relationship_visualizations": [
        {{
            "columns": ["col1", "col2"],
            "viz_type": "scatter|heatmap|grouped_bar",
            "reasoning": "why this visualization",
            "styling": {{}},
            "insights_to_look_for": []
        }}
    ],
    "overall_recommendation": "Brief summary of visualization strategy"
}}

Provide your response as valid JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert data visualization consultant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            raw_content = response.choices[0].message.content
            
            # Remove markdown code blocks if present
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content.replace("```", "").strip()
            
            # Try to parse JSON
            recommendations = json.loads(raw_content)
            
            print(f"\n📥 RECEIVED FROM LLM:")
            print(f"=" * 70)
            print(f"💡 Overall Recommendation:")
            print(f"   {recommendations.get('overall_recommendation', 'N/A')}")
            print(f"\n📊 Individual Visualizations ({len(recommendations['individual_visualizations'])}):")
            for viz in recommendations['individual_visualizations']:
                print(f"   • {viz['column']} → {viz['viz_type']}")
                print(f"     Reason: {viz['reasoning'][:80]}...")
            print(f"\n🔗 Relationship Visualizations ({len(recommendations.get('relationship_visualizations', []))}):")
            for viz in recommendations.get('relationship_visualizations', []):
                print(f"   • {' vs '.join(viz['columns'])} → {viz['viz_type']}")
                print(f"     Reason: {viz['reasoning'][:80]}...")
            print(f"=" * 70)
            
            print("✅ LLM recommendations received")
            return recommendations
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            return self._fallback_recommendations(selected_columns)
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return self._fallback_recommendations(selected_columns)
    
    def _fallback_recommendations(self, selected_columns):
        """Fallback recommendations if LLM fails"""
        
        individual_viz = []
        
        for col in selected_columns:
            analysis = self.column_analysis[col]
            
            if analysis['is_numeric']:
                viz = {
                    "column": col,
                    "viz_type": "histogram",
                    "reasoning": f"Numeric column with {analysis['unique_count']} unique values",
                    "styling": {
                        "title": f"Distribution of {col}",
                        "color": "blue",
                        "xlabel": col,
                        "ylabel": "Frequency"
                    },
                    "insights_to_look_for": ["distribution shape", "outliers", "range"]
                }
            else:
                viz = {
                    "column": col,
                    "viz_type": "bar_chart",
                    "reasoning": f"Categorical column with {analysis['unique_count']} categories",
                    "styling": {
                        "title": f"{col} Distribution",
                        "color": "green",
                        "xlabel": col,
                        "ylabel": "Count"
                    },
                    "insights_to_look_for": ["most common categories", "distribution balance"]
                }
            
            individual_viz.append(viz)
        
        return {
            "individual_visualizations": individual_viz,
            "relationship_visualizations": [],
            "overall_recommendation": "Basic visualizations for selected columns"
        }
    
    def analyze_and_recommend(self, df, selected_columns):
        """Main method: Analyze columns and get recommendations"""
        
        print(f"\n🎨 INTELLIGENT VISUALIZATION ANALYSIS")
        print(f"=" * 50)
        print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"🎯 Selected columns: {selected_columns}")
        
        # Step 1: Analyze each selected column
        print(f"\n🔍 Analyzing column characteristics...")
        for col in selected_columns:
            self.column_analysis[col] = self.analyze_column_characteristics(df, col)
            analysis = self.column_analysis[col]
            print(f"   ✓ {col}: {analysis['dtype']}, {analysis['unique_count']} unique, "
                  f"{'numeric' if analysis['is_numeric'] else 'categorical'}")
        
        # Step 2: Get LLM recommendations
        print(f"\n🤖 Getting intelligent recommendations from LLM...")
        self.recommendations = self.get_llm_visualization_recommendations(df, selected_columns)
        
        # Step 3: Show recommendations
        print(f"\n💡 VISUALIZATION RECOMMENDATIONS:")
        print(f"   {self.recommendations.get('overall_recommendation', 'N/A')}")
        
        return self.recommendations
    
    def save_recommendations(self, filepath="visualization_recommendations.json"):
        """Save recommendations to file"""
        output = {
            'column_analysis': self.column_analysis,
            'recommendations': self.recommendations
        }
        
        with open(filepath, 'w') as f:
            json.dumps(output, f, indent=2)
        
        print(f"💾 Recommendations saved to {filepath}")


def test_with_sample_data():
    """Test with sample dataframe"""
    
    # Create sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'price': np.random.normal(100, 20, 1000),
        'quantity': np.random.poisson(5, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'rating': np.random.uniform(1, 5, 1000),
        'age_group': np.random.choice(['18-25', '26-35', '36-45', '45+'], 1000)
    })
    
    print("📊 SAMPLE DATASET CREATED")
    print(df.head())
    print(f"\nShape: {df.shape}")
    
    # Initialize visualizer
    visualizer = IntelligentVisualizer()
    
    # Simulate user selecting columns
    print("\n" + "=" * 50)
    print("🎯 USER COLUMN SELECTION")
    print("=" * 50)
    selected_columns = ['price', 'category', 'rating']
    print(f"Selected columns: {selected_columns}")
    
    # Get recommendations
    recommendations = visualizer.analyze_and_recommend(df, selected_columns)
    
    # Show detailed recommendations
    print(f"\n📋 DETAILED RECOMMENDATIONS:")
    print(json.dumps(recommendations, indent=2))
    
    return visualizer, df, recommendations


if __name__ == "__main__":
    test_with_sample_data()
