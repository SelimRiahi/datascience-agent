import pandas as pd
import numpy as np
import json
import ast
import warnings
from openai import OpenAI
from .secrets_config import AZURE_API_KEY, AZURE_ENDPOINT, MODEL_NAME

# Suppress the specific SettingWithCopyWarning that we're addressing
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

class CompleteDataCleaningAgent:
    def __init__(self):
        # Initialize Azure OpenAI client
        # Use endpoint as-is if it already has the correct format
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
        self.execution_log = []
        
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
            print(f"❌ API connection failed: {str(e)}")
            print("⚠️  This may cause issues with semantic analysis...")
    
    # ==========================================
    # STEP 1: SEMANTIC UNDERSTANDING
    # ==========================================
    
    def get_semantic_understanding(self, df, sample_size=5):
        """Step 1: Get semantic understanding from small sample"""
        print("🔍 Step 1: Understanding column semantics from sample...")
        
        semantic_results = {}
        
        for column in df.columns:
            # Get small sample for semantic analysis only
            col_data = df[column]
            sample_info = {
                'column_name': column,
                'data_type': str(col_data.dtype),
                'sample_values': col_data.dropna().head(sample_size).tolist(),
                'unique_': col_data.nunique(),
                'has_nulls': col_data.isnull().any()
            }
            
            # Get semantic type from LLM
            semantic_type = self.determine_semantic_type(sample_info)
            semantic_results[column] = semantic_type
            
            print(f"'{column}' → {semantic_type}")
        
        return semantic_results
    
    def determine_semantic_type(self, sample_info):
        """Determine what a column represents from small sample"""
        prompt = f"""
Analyze this column sample and determine what it represents semantically.

COLUMN SAMPLE:
{json.dumps(sample_info, indent=2, default=str)}

Based on the column name, data type, and sample values, determine what this column represents.

Return ONLY a JSON response:
{{
    "semantic_type": "Clear description of what this column represents (e.g., 'customer_age', 'employee_salary', 'product_rating', 'user_email', 'phone_number', 'order_date', 'product_category', etc.)"
}}

Be specific and descriptive. Use business/domain terms that clearly indicate the column's purpose.
Respond with valid JSON only.
"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Clean and parse response
            if response_text.startswith('```json'):
                response_text = response_text[7:].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:].strip()
            if response_text.endswith('```'):
                response_text = response_text[:-3].strip()
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]
            
            result = json.loads(response_text)
            return result.get('semantic_type', 'unknown_column_type')
            
        except Exception as e:
            print(f"⚠️  Error determining semantic type for '{sample_info.get('column_name', 'unknown')}': {str(e)}")
            return f"unknown_column_type"
    
    # ==========================================
    # STEP 2: LOGIC-BASED RECOMMENDATIONS
    # ==========================================
    
    def get_cleaning_logic_for_semantic_type(self, semantic_type):
        """Step 2: Get cleaning logic based on semantic understanding"""
        
        prompt = f"""
You are a data cleaning expert. Based on the semantic type of a column, suggest logical cleaning rules and actions.

COLUMN SEMANTIC TYPE: "{semantic_type}"

Provide 2-3 focused cleaning suggestions with practical alternatives.

Return a JSON response:
{{
    "business_rules": [
        "List of business/logical rules that should apply to this type of data"
    ],
    "cleaning_suggestions": [
        {{
            "action": "Primary cleaning action (e.g., 'remove_negative_values', 'cap_extreme_outliers')",
            "description": "Clear description of what to do",
            "priority": "critical|high|medium|low",
            "justification": "Why this rule makes business sense",
            "automated": true/false,
            "alternative_actions": [
                {{
                    "action": "fill_with_mean",
                    "description": "Replace problematic values with column mean"
                }},
                {{
                    "action": "mark_as_missing", 
                    "description": "Mark problematic values as missing for review"
                }}
            ]
        }}
    ]
}}

IMPORTANT GUIDELINES:
1. Provide only 2-3 FOCUSED suggestions per column type
2. Keep alternative actions to 2-3 practical options (not 6)
3. Use conservative thresholds (3-5x std dev, not extreme multipliers)
4. Focus on the most common data quality issues for this semantic type

EXAMPLES BY TYPE:
- Numeric (age, salary, rating): negative values, extreme outliers
- Text (names, emails): formatting, invalid patterns  
- Identifiers: duplicates, missing values

Respond with valid JSON only.
"""
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a data cleaning expert. Provide focused, practical suggestions with 2-3 alternatives maximum. Respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Clean response
            if response_text.startswith('```json'):
                response_text = response_text[7:].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:].strip()
            if response_text.endswith('```'):
                response_text = response_text[:-3].strip()
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]
            
            logic = json.loads(response_text)
            return logic
            
        except Exception as e:
            print(f"⚠️  Error getting cleaning logic for '{semantic_type}': {str(e)}")
            return {
                "business_rules": ["Could not determine rules"],
                "cleaning_suggestions": []
            }
    
    def generate_cleaning_recommendations(self, df, semantic_types):
        """Generate cleaning recommendations based on semantic types"""
        print(f"\n🧠 Step 2: Generating cleaning logic for each column type...")
        
        cleaning_recommendations = {}
        
        for column, semantic_type in semantic_types.items():
            print(f"Generating logic for '{column}' ({semantic_type})...")
            
            cleaning_logic = self.get_cleaning_logic_for_semantic_type(semantic_type)
            
            cleaning_recommendations[column] = {
                "semantic_type": semantic_type,
                "cleaning_logic": cleaning_logic
            }
        
        return {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": df.columns.tolist()
            },
            "semantic_mapping": semantic_types,
            "cleaning_recommendations": cleaning_recommendations
        }
    
    # ==========================================
    # STEP 3: DYNAMIC FUNCTION GENERATION
    # ==========================================
    
    def generate_cleaning_function(self, action_details, df_info):
        """Generate Python function code for a specific cleaning action"""
        
        prompt = f"""
You are a Python data cleaning expert. Generate a Python function that performs a specific cleaning action on a pandas DataFrame.

CLEANING ACTION DETAILS:
{json.dumps(action_details, indent=2, default=str)}

DATAFRAME INFO:
{json.dumps(df_info, indent=2, default=str)}

Generate a Python function that:
1. Takes a pandas DataFrame as input
2. Performs the specific cleaning action
3. Returns the cleaned DataFrame and a log message
4. Is safe and handles edge cases
5. Handles data type compatibility properly

Return your response in this EXACT format:
```python
def cleaning_function(df):
    '''
    Description: [Brief description of what this function does]
    Target: [Column(s) being cleaned]
    Action: [Specific action being performed]
    '''
    try:
        import pandas as pd
        import numpy as np
        
        # Store original row count for logging
        original_rows = len(df)
        
        # [YOUR CLEANING LOGIC HERE]
        # Make sure to use the exact column names from df_info
        # Handle missing values and edge cases
        # Create df_cleaned as the result
        
        # Calculate impact
        rows_affected = original_rows - len(df_cleaned)
        
        # Create log message
        log_message = f"[Specific description of what was done and impact]"
        
        return df_cleaned, log_message
        
    except Exception as e:
        # Return original DataFrame if error occurs
        return df, f"Error in cleaning: {{str(e)}}"
```

CRITICAL DATA TYPE REQUIREMENTS:
- When assigning float values to integer columns, use round() function to convert to compatible type
- For fill_with_mean operations on integer columns: use round(mean_value) or int(mean_value)
- For fill_with_median operations on integer columns: use round(median_value) or int(median_value)  
- Check the original column dtype from df_info and preserve it after cleaning
- Example: If column is int64 and you calculate mean=43.67, assign round(43.67)=44
- IMPORTANT: After all cleaning operations, restore the original column dtype using .astype()
- Example: df_cleaned['age'] = df_cleaned['age'].astype('int64') if original was int64
- Handle NaN values before converting back to int: either fill them or use nullable integer type 'Int64'

DYNAMIC OUTLIER DETECTION REQUIREMENTS:
- Use iterative statistical approach without domain assumptions
- Step 1: Calculate initial Q1, Q3, and IQR for the entire dataset
- Step 2: Identify potential extreme outliers using IQR method (Q1-2*IQR, Q3+2*IQR)
- Step 3: Calculate "clean" statistics (mean, median) EXCLUDING the extreme outliers
- Step 4: Compare each value against the clean statistics:
  * If value > clean_mean + 3*clean_std OR value < clean_mean - 3*clean_std, mark as outlier
  * This makes detection relative to the actual data distribution
- Step 5: Replace outliers with the clean_mean (calculated without outliers)
- This approach adapts to any dataset without hardcoded thresholds
- Works for any numeric column regardless of what it represents
- CRITICAL: Do NOT remove rows - only replace outlier VALUES within existing rows
- Always preserve the same number of rows in the output DataFrame
- Use df.loc[condition, column] = replacement_value for in-place replacement

IMPORTANT REQUIREMENTS:
- Use ONLY pandas and numpy operations
- Use the exact column names provided in df_info
- Handle all edge cases (missing values, empty dataframe, etc.)
- Always return (dataframe, log_message) tuple
- Make the function safe - no file operations, no dangerous code
- Be specific about what the function does in the docstring
- Always respect original column data types to avoid dtype compatibility warnings
- Use data-driven outlier detection - no hardcoded domain rules
- CRITICAL: Count and report actual outliers detected and replaced
- For outlier actions: Count values that meet outlier criteria BEFORE replacement
- Log format should be: "Replaced X outlier values in 'column' with [method]" where X is actual count
- If no outliers found, log: "No outliers detected in 'column' using current criteria"

PANDAS BEST PRACTICES TO AVOID WARNINGS:
- NEVER modify slices directly - always work on the full DataFrame
- Use df_cleaned = df.copy() at the beginning to create a proper copy
- Use df_cleaned.loc[condition, column] = value instead of df_cleaned[condition][column] = value
- Use .copy() when creating filtered DataFrames: df_filtered = df[condition].copy()
- For operations that might create views, explicitly use .copy()
- Example: df_cleaned = df.dropna().copy() instead of df_cleaned = df.dropna()
- When filtering and modifying: df_cleaned.loc[df_cleaned['col'] > 5, 'col'] = replacement_value

Generate the function code only, no additional explanation.
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python data cleaning expert. Generate safe, efficient pandas code only. Return code in the exact format requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Extract Python code from response
            if '```python' in response_text:
                start_idx = response_text.find('```python') + 9
                end_idx = response_text.find('```', start_idx)
                if end_idx != -1:
                    code = response_text[start_idx:end_idx].strip()
                else:
                    code = response_text[start_idx:].strip()
            else:
                # If no markdown, try to find function definition
                lines = response_text.split('\n')
                code_lines = []
                in_function = False
                for line in lines:
                    if line.strip().startswith('def '):
                        in_function = True
                    if in_function:
                        code_lines.append(line)
                code = '\n'.join(code_lines)
            
            return code
            
        except Exception as e:
            # Return a safe fallback function
            return f'''
def cleaning_function(df):
    """
    Fallback function - no cleaning performed due to generation error
    """
    return df, f"Could not generate cleaning function: {str(e)}"
'''
    
    # ==========================================
    # STEP 4: SAFETY & EXECUTION
    # ==========================================
    
    def validate_function_safety(self, function_code):
        """Validate that generated function code is safe to execute"""
        
        # More specific dangerous patterns that are actually dangerous
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'open(', 'file(', 'exec(', 'eval(', '__import__',
            'socket', 'urllib', 'requests'
        ]
        
        # Dangerous file system operations (but allow pandas operations)
        dangerous_file_ops = [
            'os.remove', 'os.unlink', 'os.rmdir', 'shutil.rmtree',
            'rm -', 'del /', 'rmdir /'
        ]
        
        # Check for dangerous patterns - be more specific
        code_lower = function_code.lower()
        for pattern in dangerous_patterns:
            # More specific checks to avoid false positives
            if pattern in code_lower:
                # Make sure it's not in a comment or string literal
                lines = function_code.split('\n')
                for line in lines:
                    line_clean = line.strip().lower()
                    if pattern in line_clean and not (line_clean.startswith('#') or line_clean.startswith('"""') or line_clean.startswith("'''")):
                        # Further check if it's actually an import statement or dangerous call
                        if ('import ' + pattern.replace('import ', '')) in line_clean or pattern + '(' in line_clean:
                            return False, f"Potentially dangerous operation detected: {pattern}"
        
        # Check for dangerous file operations (more specific)
        for pattern in dangerous_file_ops:
            if pattern in code_lower:
                return False, f"Dangerous file operation detected: {pattern}"
        
        # Try to parse the code to ensure it's valid Python
        try:
            ast.parse(function_code)
        except SyntaxError as e:
            return False, f"Invalid Python syntax: {str(e)}"
        
        # More flexible function name check - look for any function definition
        if 'def ' not in function_code:
            return False, "No function definition found"
        
        # If there's a function, try to find its name
        lines = function_code.split('\n')
        function_found = False
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('def ') and '(' in line_stripped:
                function_found = True
                break
        
        if not function_found:
            return False, "No valid function definition found"
        
        # Check for only pandas and numpy imports (more flexible)
        for line in lines:
            line_stripped = line.strip().lower()
            if line_stripped.startswith('import '):
                if not any(allowed in line_stripped for allowed in ['pandas', 'numpy', 'pd', 'np']):
                    # Allow if it's importing from standard library that's safe
                    if not any(safe in line_stripped for safe in ['import re', 'import datetime', 'import math']):
                        return False, f"Potentially unsafe import: {line.strip()}"
        
        return True, "Function appears safe"
    
    def preview_cleaning_action(self, df, action_details, sample_size=5):
        """Preview what a cleaning action will do without executing it"""
        
        print(f"🔍 PREVIEW: {action_details.get('action', 'Unknown Action')}")
        print("-" * 40)
        
        # Generate function
        df_info = {
            "columns": df.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "sample_rows": min(len(df), 5)
        }
        
        function_code = self.generate_cleaning_function(action_details, df_info)
        
        # Validate safety
        is_safe, safety_message = self.validate_function_safety(function_code)
        
        print(f"📋 Action: {action_details.get('description', 'No description')}")
        print(f"🎯 Target: {action_details.get('target', 'Unknown target')}")
        print(f"🔒 Safety Check: {'✅ SAFE' if is_safe else '❌ UNSAFE - ' + safety_message}")
        
        if not is_safe:
            return None, safety_message
        
        # Test on small sample
        try:
            sample_df = df.head(sample_size).copy()
            
            # Execute function safely
            local_scope = {'pd': pd, 'np': np}
            exec(function_code, local_scope)
            
            # Find the function that was defined (more flexible)
            function_names = [name for name in local_scope if callable(local_scope[name]) and name != 'pd' and name != 'np']
            if not function_names:
                return None, "No cleaning function found in generated code"
            
            cleaning_function = local_scope[function_names[0]]
            
            # Test the function
            result_df, log_message = cleaning_function(sample_df)
            
            print(f"📊 Sample Test Results:")
            print(f"   Original rows: {len(sample_df)}")
            print(f"   Result rows: {len(result_df)}")
            print(f"   Log: {log_message}")
            
            # Show code preview (first few lines)
            code_lines = function_code.split('\n')[:10]
            print(f"\n💻 Generated Function Preview:")
            for line in code_lines:
                print(f"   {line}")
            if len(function_code.split('\n')) > 10:
                print("   ... (truncated)")
            
            return function_code, "Preview successful"
            
        except Exception as e:
            error_msg = f"Preview failed: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
    
    def execute_cleaning_action(self, df, action_details, approved_function_code):
        """Execute a cleaning action on the full dataframe"""
        
        # Final safety check
        is_safe, safety_message = self.validate_function_safety(approved_function_code)
        if not is_safe:
            return df, f"Safety check failed: {safety_message}"
        
        try:
            # Execute the function
            local_scope = {'pd': pd, 'np': np}
            exec(approved_function_code, local_scope)
            
            # Find the function that was defined (more flexible)
            function_names = [name for name in local_scope if callable(local_scope[name]) and name != 'pd' and name != 'np']
            if not function_names:
                return df, "No cleaning function found in generated code"
            
            # Use the first function found
            cleaning_function = local_scope[function_names[0]]
            
            # Apply to full dataframe
            result_df, log_message = cleaning_function(df)
            
            # Log the execution
            execution_record = {
                "action": action_details.get('action', 'unknown'),
                "target": action_details.get('target', 'unknown'),
                "original_rows": len(df),
                "result_rows": len(result_df),
                "rows_affected": len(df) - len(result_df),
                "log_message": log_message
            }
            self.execution_log.append(execution_record)
            
            return result_df, log_message
            
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            return df, error_msg
    
    # ==========================================
    # MAIN PIPELINE: ALL STEPS INTEGRATED
    # ==========================================
    
    def clean_dataset(self, df, sample_size=5, interactive=True):
        """Complete data cleaning pipeline - all steps integrated"""
        
        print("🤖 COMPLETE DATA CLEANING AGENT")
        print(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
        print("=" * 60)
        
        # Step 1: Semantic Understanding
        semantic_types = self.get_semantic_understanding(df, sample_size)
        
        # Step 2: Generate Cleaning Recommendations
        recommendations = self.generate_cleaning_recommendations(df, semantic_types)
        
        # Step 3: Execute Cleaning Actions
        print(f"\n🚀 Step 3: Executing cleaning actions...")
        print("=" * 50)
        
        current_df = df.copy()
        
        for column, rec in recommendations["cleaning_recommendations"].items():
            semantic_type = rec["semantic_type"]
            suggestions = rec["cleaning_logic"].get("cleaning_suggestions", [])
            
            if not suggestions:
                continue
                
            print(f"\n📋 Processing column: '{column}' ({semantic_type})")
            
            # Flag to skip entire column
            skip_column = False
            
            for suggestion in suggestions:
                # If skip_column flag is set, break out of suggestions loop
                if skip_column:
                    break
                action_details = {
                    "action": suggestion.get("action"),
                    "description": suggestion.get("description"),
                    "target": column,
                    "semantic_type": semantic_type,
                    "priority": suggestion.get("priority"),
                    "automated": suggestion.get("automated", False)
                }
                
                print(f"\n🔧 Action: {action_details['action']} ({action_details.get('priority', 'unknown')} priority)")
                
                # Show alternatives upfront (simplified)
                alternatives = suggestion.get("alternative_actions", [])
                if alternatives:
                    print("🔄 Options:")
                    print(f"   0. {action_details['action']}: {action_details.get('description', 'Primary action')}")
                    for i, alt in enumerate(alternatives[:3], 1):  # Limit to 3 alternatives
                        print(f"   {i}. {alt.get('action', 'unknown')}: {alt.get('description', 'No description')}")
                
                # Preview the primary action
                function_code, preview_result = self.preview_cleaning_action(current_df, action_details)
                
                if function_code is None:
                    print(f"❌ Skipping action due to error: {preview_result}")
                    continue
                
                # Enhanced interactive choice with individual skip option
                if interactive:
                    while True:
                        if alternatives:
                            print(f"\nOptions: 0-{min(len(alternatives), 3)}, 'skip'=skip this action, 'skip_col'=skip entire column, 'm'=multiple, 's'=skip all")
                            choice = input("Choose: ").lower().strip()
                            
                            if choice == 's' or choice == 'skip_all':
                                print("⏭️  Skipping all remaining actions")
                                return current_df, recommendations
                            elif choice == 'skip':
                                print("⏭️  Skipping this action")
                                function_code = None
                                break
                            elif choice == 'skip_col' or choice == 'skip_column':
                                print(f"⏭️  Skipping entire column '{column}' - moving to next column")
                                function_code = None
                                # Set a flag to break out of the suggestions loop for this column
                                skip_column = True
                                break
                            elif choice == 'm' or choice == 'multiple':
                                # Multi-selection mode
                                print("📋 Multi-selection mode:")
                                print("Enter numbers separated by spaces (e.g., '0 2' for primary + alternative 2)")
                                multi_choice = input("Choices: ").strip()
                                
                                try:
                                    choices = [int(x) for x in multi_choice.split() if x.isdigit()]
                                    valid_choices = [c for c in choices if 0 <= c <= min(len(alternatives), 3)]
                                    
                                    if valid_choices:
                                        print(f"✅ Selected actions: {valid_choices}")
                                        
                                        # Execute multiple actions in sequence
                                        for choice_num in valid_choices:
                                            if choice_num == 0:
                                                # Primary action
                                                execute_action = action_details.copy()
                                            else:
                                                # Alternative action
                                                chosen_alt = alternatives[choice_num - 1]
                                                execute_action = action_details.copy()
                                                execute_action["action"] = chosen_alt.get("action")
                                                execute_action["description"] = chosen_alt.get("description")
                                            
                                            print(f"\n⚡ Executing: {execute_action['action']}")
                                            
                                            # Check if we still have data to clean
                                            if len(current_df) == 0:
                                                print(f"❌ No data left to clean - skipping {execute_action['action']}")
                                                continue
                                            
                                            # Generate and execute function
                                            multi_function_code = self.generate_cleaning_function(
                                                execute_action, 
                                                {"columns": current_df.columns.tolist(), "dtypes": {str(k): str(v) for k, v in current_df.dtypes.to_dict().items()}}
                                            )
                                            
                                            if self.validate_function_safety(multi_function_code)[0]:
                                                current_df, log_message = self.execute_cleaning_action(current_df, execute_action, multi_function_code)
                                                print(f"✅ {log_message}")
                                                
                                                # Show current dataset status
                                                print(f"   📊 Current dataset: {len(current_df)} rows × {len(current_df.columns)} columns")
                                            else:
                                                print(f"❌ Skipped {execute_action['action']} due to safety issues")
                                        
                                        function_code = None  # Skip normal execution
                                        break
                                    else:
                                        print("❌ No valid choices selected")
                                        
                                except ValueError:
                                    print("❌ Invalid input format. Use numbers separated by spaces.")
                                    
                            elif choice == '0':
                                # Use primary action
                                break
                            elif choice.isdigit() and 1 <= int(choice) <= min(len(alternatives), 3):
                                # Use alternative action
                                chosen_alt = alternatives[int(choice) - 1]
                                action_details["action"] = chosen_alt.get("action")
                                action_details["description"] = chosen_alt.get("description")
                                print(f"✅ Using: {chosen_alt.get('action')}")
                                
                                # Preview the alternative action
                                function_code, preview_result = self.preview_cleaning_action(current_df, action_details)
                                if function_code:
                                    break
                                else:
                                    print(f"❌ Action failed: {preview_result}")
                                    function_code = None
                                    break
                            else:
                                print(f"Please enter 0-{min(len(alternatives), 3)}, 'skip', 'm' for multiple, or 's' to skip all")
                        else:
                            # No alternatives, simple y/n/skip/skip_col/s choice
                            choice = input(f"\nExecute this cleaning action? (y/n/skip=skip this/skip_col=skip column/s=skip all): ").lower().strip()
                            if choice in ['y', 'yes']:
                                break
                            elif choice in ['n', 'no', 'skip']:
                                print("⏭️  Skipping this action")
                                function_code = None
                                break
                            elif choice in ['skip_col', 'skip_column']:
                                print(f"⏭️  Skipping entire column '{column}' - moving to next column")
                                function_code = None
                                skip_column = True
                                break
                            elif choice in ['s', 'skip_all']:
                                print("⏭️  Skipping all remaining actions")
                                return current_df, recommendations
                            else:
                                print("Please enter 'y' for yes, 'n' or 'skip' to skip this action, 'skip_col' to skip entire column, or 's' to skip all")
                
                # Execute if approved and we still have data
                if function_code and len(current_df) > 0:
                    print(f"⚡ Executing cleaning action...")
                    current_df, log_message = self.execute_cleaning_action(current_df, action_details, function_code)
                    print(f"✅ {log_message}")
                    
                    # Show dataset status after each action
                    print(f"   📊 Dataset status: {len(current_df)} rows × {len(current_df.columns)} columns")
                    
                    # Show a sample of current data if not empty
                    if len(current_df) > 0:
                        print(f"   📋 Sample data:")
                        print(f"   {current_df.head(2).to_string()}")
                elif len(current_df) == 0:
                    print(f"❌ Cannot execute - no data remaining in dataset!")
                    break
        
        return current_df, recommendations
    
    def print_execution_summary(self):
        """Print summary of all executed cleaning actions"""
        
        print("\n" + "="*60)
        print("📊 CLEANING EXECUTION SUMMARY")
        print("="*60)
        
        if not self.execution_log:
            print("No cleaning actions were executed.")
            return
        
        total_rows_affected = sum(record["rows_affected"] for record in self.execution_log)
        
        print(f"Total actions executed: {len(self.execution_log)}")
        print(f"Total rows affected: {total_rows_affected}")
        
        for i, record in enumerate(self.execution_log, 1):
            print(f"\n{i}. {record['action']} on '{record['target']}'")
            print(f"   Rows: {record['original_rows']} → {record['result_rows']} ({record['rows_affected']} affected)")
            print(f"   Result: {record['log_message']}")

# Example usage
if __name__ == "__main__":
    def create_sample_problematic_dataframe():
        """Create sample data with various issues for testing"""
        np.random.seed(42)
        
        data = {
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P001'],  # Duplicate ID
            'product_name': ['Laptop Pro', 'Wireless Mouse', '', 'Gaming Keyboard', None, 'Monitor 4K', 'USB Cable', 'Tablet Mini', 'Smartphone X', 'Laptop Pro'],
            'price': [1299.99, 29.95, 89.99, -50.00, 599.99, 25000.00, 12.99, 349.99, 899.99, 1299.99],  # Negative price, extreme high price
            'rating': [4.5, 3.8, 2.1, 4.9, -1.0, 15.5, 4.2, 3.6, 4.8, 4.5],  # Negative rating, rating > 5
            'stock_quantity': [50, 120, 0, 75, -10, 999999, 25, 45, 88, 50],  # Negative stock, extremely high stock
            'category': ['Electronics', 'Accessories', 'Electronics', 'Accessories', None, 'Electronics', '', 'Electronics', 'Electronics', 'Electronics'],
            'supplier_email': ['tech@supplier.com', 'sales@mouse.co', 'invalid-email-format', 'gaming@keys.com', None, 'monitor@display.net', 'cable@usb', 'tablet@mini.com', 'phone@smart.co', 'tech@supplier.com'],
            'launch_date': ['2023-01-15', '2022-05-20', 'invalid-date', '2023-03-10', '2021-12-25', '2025-12-31', '', '2022-08-05', '2023-09-15', '2023-01-15'],  # Invalid date, future date
            'weight_kg': [2.1, 0.15, 0.8, 1.2, 0.0, -5.5, 0.05, 0.65, 0.18, 2.1],  # Zero weight, negative weight
            'warranty_months': [24, 12, 36, 18, 0, 240, 6, 24, 12, 24]  # Extremely long warranty
        }
        
        return pd.DataFrame(data)
    
    # Load data from Excel file
    data_path = r"C:\Users\Selim\OneDrive\Bureau\data science agent\data\data.xlsx"
    
    try:
        df = pd.read_excel(data_path)
        print("🔍 Successfully loaded data from Excel file!")
    except FileNotFoundError:
        print(f"❌ File not found: {data_path}")
        print("🔄 Using sample data instead...")
        df = create_sample_problematic_dataframe()
    except Exception as e:
        print(f"❌ Error loading Excel file: {str(e)}")
        print("🔄 Using sample data instead...")
        df = create_sample_problematic_dataframe()
    
    print("🔍 Original Dataset:")
    print(df)
    print(f"Shape: {df.shape}")
    
    # Create and run the complete agent
    agent = CompleteDataCleaningAgent()
    
    # Run the complete cleaning pipeline
    cleaned_df, recommendations = agent.clean_dataset(df, sample_size=3, interactive=True)
    
    print(f"\n🎉 FINAL RESULTS:")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print("\nCleaned Dataset:")
    print(cleaned_df)
    
    # Show execution summary
    agent.print_execution_summary()
    
    # Proper Excel formatting - ensure clean data structure
    print(f"\n🔧 Formatting data for Excel export...")
    
    # Clean up any problematic characters and ensure proper data types
    formatted_df = cleaned_df.copy()
    
    for column in formatted_df.columns:
        if formatted_df[column].dtype == 'object':
            # Clean text columns - remove extra spaces and newlines for Excel
            formatted_df[column] = formatted_df[column].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
            formatted_df[column] = formatted_df[column].str.strip()  # Remove leading/trailing spaces
            
            # Handle None/nan values properly
            formatted_df[column] = formatted_df[column].replace(['nan', 'None', 'NaN'], '')
    
    # Save results with proper Excel formatting
    formatted_df.to_excel('cleaned_dataset.xlsx', index=False, engine='openpyxl')
    print(f"✅ Properly formatted Excel file saved!")
    
    with open('cleaning_results.json', 'w') as f:
        results = {
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "recommendations": recommendations,
            "execution_log": agent.execution_log
        }
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Cleaned dataset saved to 'cleaned_dataset.xlsx'")
    print(f"💾 Full results saved to 'cleaning_results.json'")