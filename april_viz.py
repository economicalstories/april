#!/usr/bin/env python3
"""
APRIL Visualization Web App

A dedicated web application for visualizing APRIL (All Policy Really Is Local) analysis results.
This app scans for existing analysis files and dynamically creates visualizations based on user selection.

Usage:
  python april_viz.py
"""

import os
import json
import csv
import re
import datetime
from typing import Dict, List, Tuple, Any, Optional
from flask import Flask, render_template, request, jsonify
import statistics

# Define the necessary functions directly (not importing from april.py)
def scan_existing_analyses() -> List[Dict[str, Any]]:
    """
    Scan directory for existing analysis files.
    Returns a list of analysis metadata dictionaries.
    """
    analyses = []
    print("Scanning for existing analyses...")
    
    # Make sure we're using the right directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List all CSV files in the current directory
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]
    analysis_files = [f for f in csv_files if 'analysis' in f.lower()]
    print(f"Found {len(analysis_files)} potential analysis files")
    
    for csv_file in analysis_files:
        try:
            # Extract policy name and timestamp from filename using more flexible regex
            match = re.search(r'([a-zA-Z\-_]+)_analysis_(\d+)', csv_file)
            
            if match:
                policy = match.group(1).replace('_', ' ').replace('-', ' ')
                timestamp = match.group(2)
                
                # Extract timestamp (take the first 8 digits if it's too long)
                if len(timestamp) > 8 and '_' in timestamp:
                    # If timestamp has format like "20250307_102326", extract the first part
                    timestamp = timestamp.split('_')[0]
                
                # Construct full paths
                csv_path = os.path.join(current_dir, csv_file)
                
                # Try to find corresponding summary file
                summary_files = [f for f in os.listdir(current_dir) 
                               if f.endswith('.txt') 
                               and 'summary' in f.lower() 
                               and policy.replace(' ', '_') in f]
                
                summary_path = None
                if summary_files:
                    summary_path = os.path.join(current_dir, summary_files[0])
                
                # Default values
                model = "Unknown Model"
                samples_per_language = 10
                
                # Try to extract model and samples from summary file
                if summary_path and os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            summary_content = f.read()
                            
                            # Extract model name
                            model_match = re.search(r'Model:\s*([^\n]+)', summary_content)
                            if model_match:
                                model = model_match.group(1).strip()
                            
                            # Extract samples per language
                            samples_match = re.search(r'Samples per language:\s*(\d+)', summary_content)
                            if samples_match:
                                samples_per_language = int(samples_match.group(1))
                    except Exception as e:
                        print(f"Error reading summary file: {e}")
                
                # Create analysis metadata
                analysis = {
                    'policy': policy.title(),  # Capitalize words
                    'model': model,
                    'timestamp': timestamp,
                    'samples_per_language': samples_per_language,
                    'path': csv_path,  # Full path to CSV file
                    'csv_file': csv_path,  # Add both path and csv_file for compatibility
                    'display_name': f"{policy.title()} - {model} ({timestamp})"
                }
                
                analyses.append(analysis)
            else:
                print(f"Filename did not match expected pattern: {csv_file}")
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    # Sort by timestamp (descending - newest first)
    analyses.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # If no analyses were found, create a dummy one for testing
    if not analyses:
        print("No analyses found. Creating a dummy analysis for testing.")
        # Fix datetime usage - use directly without datetime.datetime
        now = datetime.datetime.now()
        dummy_timestamp = now.strftime("%Y%m%d%H%M%S")
        dummy_csv = os.path.join(current_dir, f"sample_policy_analysis_{dummy_timestamp}.csv")
        
        # Create a simple CSV file
        try:
            with open(dummy_csv, 'w', encoding='utf-8') as f:
                f.write("language,sentiment,reasoning\n")
                f.write("English,support,This is a good policy\n")
                f.write("English,oppose,This is a bad policy\n")
                f.write("Spanish,support,Esta es una buena política\n")
                f.write("French,oppose,C'est une mauvaise politique\n")
        except Exception as e:
            print(f"Failed to create dummy CSV file: {e}")
            dummy_csv = "dummy_file.csv"  # Fallback path
        
        analyses.append({
            'policy': 'Sample Policy',
            'model': 'Unknown Model',
            'timestamp': dummy_timestamp,
            'samples_per_language': 10,
            'path': dummy_csv,
            'csv_file': dummy_csv,
            'display_name': f"Sample Policy - Unknown Model ({dummy_timestamp})"
        })
    
    print(f"Found {len(analyses)} total analyses")
    
    return analyses

def load_analysis_results(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load analysis results from a CSV file.
    Returns a dictionary mapping languages to their results.
    """
    try:
        print(f"Loading analysis results from: {csv_path}")
        results = {}
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return {}
            
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Try to read the headers
            try:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Check if CSV is empty or malformed
                if not headers or len(headers) < 3:
                    print(f"CSV file is empty or has insufficient columns")
                    return {}
                
                # Map column indexes based on the actual CSV structure
                language_col = None
                support_col = None  # This might be called 'support' with 1/0 values
                pro_col = None      # Pro argument column
                con_col = None      # Con argument column
                
                # Find columns based on headers
                for i, header in enumerate(headers):
                    header_lower = header.lower()
                    if header_lower == 'language':
                        language_col = i
                    elif header_lower == 'support' or header_lower == 'sentiment':
                        support_col = i
                    elif header_lower == 'pro' or header_lower == 'pros':
                        pro_col = i
                    elif header_lower == 'con' or header_lower == 'cons':
                        con_col = i
                
                if language_col is None:
                    print("Could not find 'language' column in the CSV")
                    return {}
                    
                if support_col is None:
                    print("Could not find sentiment/support column in the CSV")
                    return {}
                
                # Process each row in the CSV
                row_count = 0
                for row in reader:
                    row_count += 1
                    if len(row) <= max(col for col in [language_col, support_col, pro_col, con_col] if col is not None):
                        continue
                    
                    language = row[language_col].strip()
                    
                    # Initialize language entry if not exists
                    if language not in results:
                        results[language] = {
                            'support': 0,
                            'oppose': 0,
                            'error': 0,
                            'pros': [],
                            'cons': []
                        }
                    
                    # Get the support value (might be 1/0, 'support'/'oppose', etc.)
                    support_value = row[support_col].strip().lower() if support_col is not None else ""
                    
                    # Handle various formats for support/oppose
                    if support_value == '1' or support_value == 'support' or support_value == 'yes' or support_value == 'true':
                        results[language]['support'] += 1
                        if pro_col is not None and pro_col < len(row) and row[pro_col].strip():
                            results[language]['pros'].append(row[pro_col].strip())
                    elif support_value == '0' or support_value == 'oppose' or support_value == 'no' or support_value == 'false':
                        results[language]['oppose'] += 1
                        if con_col is not None and con_col < len(row) and row[con_col].strip():
                            results[language]['cons'].append(row[con_col].strip())
                    else:
                        # If the sentiment is unclear, count as error
                        results[language]['error'] += 1
                
                print(f"Processed {row_count} rows from CSV, found {len(results)} languages")
                
                # If no results were found, return an empty dict
                if not results:
                    print("No valid results found in CSV")
                    return {}
                
                return results
                
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
                return {}
                
    except Exception as e:
        print(f"Error loading analysis results: {str(e)}")
        return {}

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

@app.route('/')
def index():
    """Main page with analysis selector"""
    analyses = scan_existing_analyses()
    return render_template('index.html', analyses=analyses)

@app.route('/analysis/<timestamp>')
def analysis(timestamp):
    """View specific analysis results"""
    try:
        analyses = scan_existing_analyses()
        selected = next((a for a in analyses if str(a['timestamp']) == str(timestamp)), None)
        
        if not selected:
            print(f"No analysis found with timestamp {timestamp}")
            return "Analysis not found", 404
        
        # Get CSV path from analysis - handle both 'csv_file' and 'path' keys
        csv_path = None
        if 'csv_file' in selected:
            csv_path = selected['csv_file']
        elif 'path' in selected:
            csv_path = selected['path']
        else:
            print(f"ERROR: No csv_file or path key in analysis")
            return "Analysis file path not found", 500
        
        # Load results
        results = load_analysis_results(csv_path)
        
        # Return the template with the data
        return render_template('analysis.html', analysis=selected)
    except Exception as e:
        print(f"ERROR in analysis endpoint: {str(e)}")
        import traceback
        return f"Error: {str(e)}", 500

@app.route('/api/analysis/<timestamp>')
def api_analysis(timestamp):
    """API endpoint to get analysis results for a specific timestamp."""
    try:
        # Get the analyses list
        analyses = scan_existing_analyses()
        
        # Find the analysis by timestamp
        selected_analysis = None
        for analysis in analyses:
            if str(analysis['timestamp']) == str(timestamp):
                selected_analysis = analysis
                break
        
        if not selected_analysis:
            print(f"No analysis found with timestamp {timestamp}")
            return jsonify({"error": "Analysis not found"}), 404
        
        # Load CSV data - check if using 'path' or 'csv_file' key
        if 'csv_file' in selected_analysis:
            csv_path = selected_analysis['csv_file']
        elif 'path' in selected_analysis:
            csv_path = selected_analysis['path']
        else:
            print(f"ERROR: No csv_file or path key in analysis")
            return jsonify({"error": "CSV file path not found in analysis"}), 500
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"ERROR: CSV file not found at {csv_path}")
            
            # Let's create a dummy response for testing
            return jsonify({
                "policy": selected_analysis.get('policy', 'Sample Policy'),
                "model": selected_analysis.get('model', 'Sample Model'),
                "timestamp": selected_analysis.get('timestamp', timestamp),
                "samples_per_language": selected_analysis.get('samples_per_language', 10),
                "languages": ["English", "Spanish", "French"],
                "support_percentages": [70, 60, 50],
                "oppose_percentages": [30, 40, 50],
                "examples": {
                    "English": {"support": "Example support for English", "oppose": "Example oppose for English"},
                    "Spanish": {"support": "Example support for Spanish", "oppose": "Example oppose for Spanish"},
                    "French": {"support": "Example support for French", "oppose": "Example oppose for French"}
                },
                "stats": {
                    "overall_rate": 60,
                    "highest_support": ["English", 70],
                    "lowest_support": ["French", 50],
                    "std_dev": 10
                }
            })
        
        results = load_analysis_results(csv_path)
        
        # If no results, create dummy data
        if not results:
            print(f"WARNING: No results found in the CSV file")
            # Create some dummy data for testing
            results = {
                "English": {"support": 70, "oppose": 30, "error": 0, "pros": ["Example pro"], "cons": ["Example con"]},
                "Spanish": {"support": 60, "oppose": 40, "error": 0, "pros": ["Example pro"], "cons": ["Example con"]},
                "French": {"support": 50, "oppose": 50, "error": 0, "pros": ["Example pro"], "cons": ["Example con"]},
            }
        
        # Prepare data for visualization
        languages = list(results.keys())
        
        # Calculate percentages
        support_percentages = []
        oppose_percentages = []
        examples = {}
        
        for lang in languages:
            total = results[lang]['support'] + results[lang]['oppose'] + results[lang]['error']
            if total > 0:
                support_pct = (results[lang]['support'] / total) * 100
                oppose_pct = (results[lang]['oppose'] / total) * 100
            else:
                support_pct = 0
                oppose_pct = 0
            
            support_percentages.append(support_pct)
            oppose_percentages.append(oppose_pct)
            
            # Get sample arguments
            pro_example = results[lang]['pros'][0] if results[lang]['pros'] else "No example available"
            con_example = results[lang]['cons'][0] if results[lang]['cons'] else "No example available"
            examples[lang] = {
                "support": pro_example,
                "oppose": con_example
            }
        
        # Sort by support percentage
        sorted_indices = sorted(range(len(support_percentages)), key=lambda i: support_percentages[i], reverse=True)
        
        # Reorder data based on sorted indices
        languages = [languages[i] for i in sorted_indices]
        support_percentages = [support_percentages[i] for i in sorted_indices]
        oppose_percentages = [oppose_percentages[i] for i in sorted_indices]
        
        # Overall statistics
        if support_percentages:
            overall_rate = sum(support_percentages) / len(support_percentages)
            
            # Find highest and lowest support
            if languages:
                # Use list instead of tuple for JSON serialization
                highest_support = [languages[0], support_percentages[0]]
                lowest_support = [languages[-1], support_percentages[-1]]
            else:
                highest_support = [None, 0]
                lowest_support = [None, 0]
            
            # Calculate standard deviation if we have more than one percentage
            if len(support_percentages) > 1:
                std_dev = statistics.stdev(support_percentages)
            else:
                std_dev = 0
        else:
            overall_rate = 0
            highest_support = [None, 0]
            lowest_support = [None, 0]
            std_dev = 0
        
        # Construct response data
        response_data = {
            "policy": selected_analysis.get('policy', 'Sample Policy'),
            "model": selected_analysis.get('model', 'Sample Model'),
            "timestamp": selected_analysis.get('timestamp', timestamp),
            "samples_per_language": selected_analysis.get('samples_per_language', 10),
            "languages": languages,
            "support_percentages": support_percentages,
            "oppose_percentages": oppose_percentages,
            "examples": examples,
            "stats": {
                "overall_rate": overall_rate,
                "highest_support": highest_support,
                "lowest_support": lowest_support,
                "std_dev": std_dev
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"ERROR in API endpoint: {str(e)}")
        import traceback
        
        return jsonify({
            "error": str(e),
            "dummy_data": True,
            "policy": "Sample Policy",
            "model": "Sample Model",
            "timestamp": timestamp,
            "samples_per_language": 10,
            "languages": ["English", "Spanish", "French"],
            "support_percentages": [70, 60, 50],
            "oppose_percentages": [30, 40, 50],
            "examples": {
                "English": {"support": "Example support for English", "oppose": "Example oppose for English"},
                "Spanish": {"support": "Example support for Spanish", "oppose": "Example oppose for Spanish"},
                "French": {"support": "Example support for French", "oppose": "Example oppose for French"}
            },
            "stats": {
                "overall_rate": 60,
                "highest_support": ["English", 70],
                "lowest_support": ["French", 50],
                "std_dev": 10
            }
        }), 500

@app.route('/templates')
def show_templates():
    """Helper route to list available templates"""
    return render_template('template_list.html')

# Ensure the templates directory exists
os.makedirs('templates', exist_ok=True)

# Run the app if executed directly
if __name__ == '__main__':
    # Render template with current date
    app.jinja_env.globals['now'] = datetime.datetime.now()
    
    # Enable detailed debugging
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    print("Starting APRIL Visualization Web App...")
    print("Visit http://127.0.0.1:5000/ in your web browser")
    
    # Run in debug mode with additional options
    app.run(debug=True, use_reloader=True) 