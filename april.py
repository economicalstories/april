#!/usr/bin/env python3
"""
Simple Policy Sentiment Analysis Tool

A streamlined tool to test how ChatGPT responds to policy questions across different languages.
This script handles the entire process in a simple workflow:
1. Ask for the policy to analyze
2. Ask which languages to test
3. Ask how many samples to generate per language
4. Run the tests and collect the results
5. Generate a CSV report and visualization of the findings

Usage:
  python simple_policy_analyzer.py
"""

import os
import json
import csv
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import glob
import sys

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Please run 'pip install openai'")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not installed. Visualizations will be disabled.")
    print("To enable visualizations, run: pip install matplotlib")

# Custom exception for aborting analysis
class AbortAnalysis(Exception):
    """Exception raised to abort analysis and proceed with collected data."""
    pass

# Dictionary of language codes for translation
LANGUAGE_OPTIONS = {
    "1": {"name": "English", "code": "english"},
    "2": {"name": "Spanish", "code": "spanish"},
    "3": {"name": "French", "code": "french"},
    "4": {"name": "German", "code": "german"},
    "5": {"name": "Chinese", "code": "chinese"},
    "6": {"name": "Japanese", "code": "japanese"},
    "7": {"name": "Korean", "code": "korean"},
    "8": {"name": "Russian", "code": "russian"},
    "9": {"name": "Arabic", "code": "arabic"},
    "10": {"name": "Hindi", "code": "hindi"},
    "11": {"name": "Portuguese", "code": "portuguese"},
    "12": {"name": "Italian", "code": "italian"}
}

# Add these model configuration constants near the top of the file, after LANGUAGE_OPTIONS
MODEL_OPTIONS = {
    "1": {
        "name": "GPT-4o",
        "id": "gpt-4o",
        "cost_per_call": 0.004,  # $0.004 per API call (estimated)
        "description": "Most powerful, highest accuracy, most expensive"
    },
    "2": {
        "name": "GPT-4o-mini",
        "id": "gpt-4o-mini",
        "cost_per_call": 0.0015,  # $0.0015 per API call (estimated)
        "description": "Good balance of capability and cost"
    },
    "3": {
        "name": "GPT-3.5-turbo",
        "id": "gpt-3.5-turbo",
        "cost_per_call": 0.0005,  # $0.0005 per API call (estimated)
        "description": "Less capable but most affordable, good for high volume"
    }
}

def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from config file or environment variable."""
    try:
        from config import OPENAI_API_KEY
        return OPENAI_API_KEY
    except ImportError:
        # Try environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        print("\nNo OpenAI API key found. You need an API key to use this tool.")
        print("Please set your API key in one of these ways:")
        print("1. Create a config.py file with: OPENAI_API_KEY = 'your-key-here'")
        print("2. Set an environment variable named OPENAI_API_KEY")
        
        # Prompt user to enter API key directly
        user_key = input("\nOr enter your OpenAI API key now: ").strip()
        if user_key:
            # Set it as an environment variable for this session
            os.environ['OPENAI_API_KEY'] = user_key
            return user_key
        
        return None

def handle_api_error(e, current_progress=None):
    """Handle API errors with user interaction."""
    error_str = str(e).lower()
    
    # Check for quota errors in multiple possible formats
    is_quota_error = (
        'insufficient_quota' in error_str or 
        'code: 429' in error_str or 
        '"code": "insufficient_quota"' in error_str or
        'exceeded your current quota' in error_str or
        '429' in error_str
    )
    
    if is_quota_error:
        print(f"\nAPI quota exceeded. Error: {e}")
        if current_progress:
            print(f"\nCurrent progress: {current_progress}")
        
        while True:
            choice = input("\nWould you like to:\n1. Wait and retry (r)\n2. Abort and analyze collected data (a)\nYour choice (r/a): ")
                if choice.lower() == 'r':
                print("\nWaiting 1 minute before retrying...")
                time.sleep(60)  # Wait for 1 minute before retrying
                return "retry"
            elif choice.lower() == 'a':
                print("\nAborting analysis completely. Saving collected data...")
                return "abort"
            else:
                print("Invalid choice. Please enter 'r' to retry or 'a' to abort.")
    return "error"

def translate_prompt(policy: str, language: str, client: Any, model_id: str) -> str:
    """Translate prompt into the target language.
    
    Args:
        policy: The name of the policy to analyze.
        language: Target language for translation.
        client: OpenAI client instance.
        model_id: The ID of the model to use.
    
    Returns:
        Translated prompt.
    """
    if language.lower() == 'english':
        return f"Explain in a sentence what {policy} is, and then indicate if you support this policy with 1 for yes or 0 for no."
    
    system_message = (
        "You are a professional translator. Translate the following English text "
        f"accurately into {language}, maintaining the same meaning and formality."
    )
    
    english_prompt = f"Explain in a sentence what {policy} is, and then indicate if you support this policy with 1 for yes or 0 for no."
    
    while True:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": english_prompt}
                ],
                temperature=0.1
            )
            
            if response and hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            else:
                print(f"Error translating to {language}. Using English prompt.")
                return english_prompt
        except Exception as e:
            action = handle_api_error(e)
            if action == "retry":
                continue  # Try again
            elif action == "abort":
                raise AbortAnalysis("Analysis aborted due to API quota error during translation")
            else:
                print(f"Error translating to {language}. Using English prompt.")
                return english_prompt

def ask_policy_question(prompt: str, client: Any, language_code: str, model_id: str) -> Dict[str, Any]:
    """
    Ask a policy question to ChatGPT and get the response.
    
    Args:
        prompt: The policy question prompt
        client: OpenAI client object
        language_code: The language code to ensure response in the same language
        model_id: The ID of the model to use
        
    Returns:
        Dictionary with explanation and binary support value
    """
    # Determine if we should ask for a non-English response
    respond_in_same_language = language_code.lower() != "english"
    
    system_message = (
        "You are a policy analyst for the government analyzing policy topics. When asked about a policy, provide:"
        "1. In one sentence, explain what the policy is, including a factual description."
        "2. In one sentence, provide the main reason to support this policy."
        "3. In one sentence, provide the main reason to oppose this policy."
        "4. State whether you ultimately support the policy (1) or oppose it (0), based solely on your analysis."
        "Format your response as a JSON object with 'explanation', 'pro', 'con', and 'support' fields."
    )
    
    if respond_in_same_language:
        system_message += f" IMPORTANT: Provide your entire response in {language_code}, not in English."
    
    # Default return in case of error
    result = {
        "explanation": None,
        "pro": None,
        "con": None,
        "support": None
    }
    
    while True:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            if response and hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
                try:
                    # Parse the JSON response
                    json_response = json.loads(response_text)
                    
                    # Extract the values
                    explanation = json_response.get("explanation")
                    pro = json_response.get("pro")
                    con = json_response.get("con")
                    support_value = json_response.get("support")
                    
                    # Convert to proper types
                    if support_value is not None:
                        try:
                            support_value = int(support_value)
                            if support_value not in [0, 1]:
                                support_value = None
                        except (ValueError, TypeError):
                            support_value = None
                    
                    return {
                        "explanation": explanation,
                        "pro": pro,
                        "con": con,
                        "support": support_value
                    }
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing response: {e}")
                    print(f"Response text: {response_text}")
                    return result
            
            print("No valid response received")
            return result
        except Exception as e:
            action = handle_api_error(e)
            if action == "retry":
                continue  # Try again
            elif action == "abort":
                print(f"\n{'='*60}")
                print("ANALYSIS ABORTED BY USER")
                print("No more API calls will be made. Proceeding with analysis of collected data.")
                print(f"{'='*60}\n")
                # Raise AbortAnalysis to break out of all loops
                raise AbortAnalysis("User requested to abort the entire analysis")
            else:
                print(f"Error asking policy question: {e}")
                return result

def create_visualization(results: Dict[str, Dict], policy: str, timestamp: str, safe_policy: str, model_name: str) -> str:
    """
    Create a bar chart visualization of the policy sentiment analysis results.
    
    Args:
        results: Dictionary containing analysis results by language
        policy: The policy topic being analyzed
        timestamp: Timestamp string for the filename
        safe_policy: Sanitized policy name for the filename
        model_name: The name of the selected model
        
    Returns:
        Path to the saved visualization file
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Visualization skipped: matplotlib is not installed.")
        return None
    
    # Prepare data for plotting
    languages = []
    support_percentages = []
    sample_sizes = []
    
    for language_name, data in results.items():
        total = data['support_count'] + data['oppose_count'] + data['error_count']
        if total > 0:
            languages.append(language_name)
            support_percentages.append((data['support_count'] / total) * 100)
            sample_sizes.append(total)
    
    # Sort by support percentage (descending)
    sorted_data = sorted(zip(languages, support_percentages, sample_sizes), 
                        key=lambda x: x[1], reverse=True)
    languages = [x[0] for x in sorted_data]
    support_percentages = [x[1] for x in sorted_data]
    sample_sizes = [x[2] for x in sorted_data]
    
    # Calculate total samples
    total_samples = sum(sample_sizes)
    samples_per_language = sample_sizes[0] if sample_sizes else 0
    
    # Create the visualization with a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a horizontal bar chart
    y_pos = range(len(languages))
    bars = ax.barh(y_pos, support_percentages, color='#2ca02c', alpha=0.8, height=0.5)
    
    # Add labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(languages, fontsize=11)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Support Percentage (%)', fontsize=11)
    
    # Set title with better formatting - now using APRIL acronym
    # Convert second line to title case
    second_line = f"{policy} Support Rates Across Languages In ChatGPT"
    second_line = ' '.join(word.capitalize() for word in second_line.split())
    title = f'APRIL: All Policy Really Is Local\n{second_line}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add metadata as a text box
    metadata = (
        f"Model: {model_name}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"Sample size: {samples_per_language} per language\n"
        f"Total API calls: {total_samples}\n"
        f"Analysis ID: {timestamp}"
    )
    
    fig.text(0.02, 0.02, metadata, fontsize=9, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add GitHub link
    github_link = "https://github.com/economicalstories/april"
    fig.text(0.5, 0.01, f"GitHub: {github_link}", ha='center', fontsize=9, 
             color='blue', alpha=0.7, fontweight='bold')
    
    # Add percentage labels on bars with conditional formatting
    for i, (bar, value) in enumerate(zip(bars, support_percentages)):
        text_color = 'black' if value < 70 else 'white'
        ax.text(max(1, value-5) if value > 10 else value + 1, 
                bar.get_y() + bar.get_height()/2, 
                f"{value:.1f}%", 
                va='center',
                ha='right' if value > 10 else 'left',
                color=text_color,
                fontweight='bold',
                fontsize=10)
    
    # Improve x-axis
    ax.set_xlim(0, max(100, max(support_percentages) + 10))
    ax.tick_params(axis='x', labelsize=10)
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.05, 0.98, 0.95])
    
    # Save the figure
    output_file = f"{safe_policy}_visualization_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_file

# Function to scan for existing analysis files
def scan_existing_analyses() -> List[Dict[str, Any]]:
    """
    Scan for existing analysis files in the current directory
    
    Returns:
        List of dictionaries containing analysis information
    """
    analyses = []
    summary_files = glob.glob("*_summary_*.txt")
    
    print(f"Looking for summary files... Found {len(summary_files)}: {summary_files}")
    
    for summary_file in summary_files:
        try:
            print(f"Processing file: {summary_file}")
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Debug - show first part of content
            print(f"File content preview: {content[:100]}...")
            
            # Updated regex to match "Policy Analysis: X" pattern
            policy_match = re.search(r"Policy(?:\s+Analysis)?:\s+(.*?)(?:\n|$)", content)
            timestamp_match = re.search(r"Analysis ID: (.*?)(?:\n|$)", content)
            languages_match = re.search(r"Number of languages: (\d+)", content)
            samples_match = re.search(r"Samples per language: (\d+)", content)
            csv_match = re.search(r"CSV file: (.*?\.csv)", content)
            model_match = re.search(r"Model: (.*?)(?:\n|$)", content)
            
            # Debug matches
            print(f"  Policy match: {policy_match.group(1) if policy_match else 'None'}")
            print(f"  Timestamp match: {timestamp_match.group(1) if timestamp_match else 'None'}")
            print(f"  Languages match: {languages_match.group(1) if languages_match else 'None'}")
            print(f"  Samples match: {samples_match.group(1) if samples_match else 'None'}")
            print(f"  CSV match: {csv_match.group(1) if csv_match else 'None'}")
            print(f"  Model match: {model_match.group(1) if model_match else 'None'}")
            
            if policy_match and timestamp_match and languages_match and samples_match and csv_match:
                policy = policy_match.group(1).strip()
                timestamp = timestamp_match.group(1).strip()
                languages_count = int(languages_match.group(1))
                samples_per_language = int(samples_match.group(1))
                csv_file = csv_match.group(1).strip()
                
                # Get model name if available, otherwise use "Unknown Model"
                model_name = "Unknown Model"
                if model_match:
                    model_name = model_match.group(1).strip()
                
                print(f"  Model name detected: {model_name}")
                
                if os.path.exists(csv_file):
            analyses.append({
                        'policy': policy,
                'timestamp': timestamp,
                        'languages': languages_count,
                        'samples_per_language': samples_per_language,
                        'csv_file': csv_file,
                        'model': model_name
                    })
                    print(f"  ✓ Successfully added analysis for {policy}")
                else:
                    print(f"  ✗ CSV file not found: {csv_file}")
            else:
                missing = []
                if not policy_match: missing.append("policy")
                if not timestamp_match: missing.append("timestamp")
                if not languages_match: missing.append("languages count") 
                if not samples_match: missing.append("samples per language")
                if not csv_match: missing.append("CSV file")
                print(f"  ✗ Missing required information: {', '.join(missing)}")
        except Exception as e:
            print(f"Error reading summary file {summary_file}: {e}")
    
    print(f"Found {len(analyses)} valid analyses")
    return analyses

# Function to load results from an existing analysis file
def load_analysis_results(csv_file: str) -> Dict[str, Dict]:
    """
    Load analysis results from a CSV file
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        Dictionary containing analysis results by language
    """
    results = {}
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Extract policy name from filename
            pattern = r'(.+)_analysis_(\d{8}_\d{6})\.csv'
            match = re.match(pattern, os.path.basename(csv_file))
            policy = match.group(1).replace('_', ' ') if match else "Unknown Policy"
            
            for row in reader:
                language = row['language']
                support = None
                
                try:
                    support = int(row['support'])
                except (ValueError, TypeError):
                    pass
                
                if language not in results:
                    results[language] = {
                        'support_count': 0,
                        'oppose_count': 0,
                        'error_count': 0,
                        'explanations': [],
                        'pros': [],
                        'cons': [],
                        'prompts': []
                    }
                
                # Update counts
                if support == 1:
                    results[language]['support_count'] += 1
                elif support == 0:
                    results[language]['oppose_count'] += 1
                else:
                    results[language]['error_count'] += 1
                
                # Store explanation, pro, con, and prompt if available
                if 'explanation' in row:
                    results[language]['explanations'].append(row['explanation'])
                if 'pro' in row:
                    results[language]['pros'].append(row['pro'])
                if 'con' in row:
                    results[language]['cons'].append(row['con'])
                if 'prompt' in row:
                    results[language]['prompts'].append(row['prompt'])
                
    except Exception as e:
        print(f"Error loading analysis results from {csv_file}: {e}")
    
    return results

# Function to visualize existing analysis
def visualize_existing_analysis(analysis_info: Dict[str, Any]) -> None:
    """
    Visualize an existing analysis
    
    Args:
        analysis_info: Dictionary containing analysis information
    """
    csv_file = analysis_info['csv_file']
    policy = analysis_info['policy']
    timestamp = analysis_info['timestamp']
    samples_per_language = analysis_info['samples_per_language']
    safe_policy = policy.replace(' ', '_').lower()
    
    print(f"\nVisualizing existing analysis for {policy}...")
    
    # Load results from CSV
    results = load_analysis_results(csv_file)
    
    # Use the model name from analysis_info
    model_name = analysis_info.get('model', "Unknown Model")
    
    # Create visualization
    if MATPLOTLIB_AVAILABLE:
        viz_file = create_visualization(results, policy, timestamp, safe_policy, model_name)
    if viz_file:
        print(f"\nVisualization created: {viz_file}")
        
        # Optionally display the visualization (if in Jupyter notebook)
        if 'ipykernel' in sys.modules:
            try:
                from IPython.display import display, Image
                display(Image(viz_file))
            except ImportError:
                pass
    
    # Create interactive HTML visualization
    interactive_html = create_interactive_html(results, policy, timestamp, safe_policy, model_name, samples_per_language)
    if interactive_html:
        print(f"Interactive HTML visualization saved as: {interactive_html}")

def check_rate_limits(client: Any, model_id: str) -> Dict[str, str]:
    """
    Make a minimal API call to check current rate limits via response headers.
    
    Args:
        client: OpenAI client instance
        model_id: The model ID to check limits for
        
    Returns:
        Dictionary containing rate limit information
    """
    print("\nChecking current API rate limits...")
    
    try:
        # Try with a minimal model first to avoid hitting quota issues
        fallback_model = "gpt-3.5-turbo" if model_id != "gpt-3.5-turbo" else "gpt-3.5-turbo-instruct"
        print(f"Using {fallback_model} for rate limit check (to conserve quota)...")
        
        # Make a minimal request to get headers
        response = client.chat.completions.create(
            model=fallback_model,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=5
        )
        
        # The OpenAI Python client doesn't directly expose headers in the response object
        # Instead, we can check if the API call succeeded as a basic connectivity test
        
        print("\n=== API Connection Test ===")
        print(f"✓ Successfully connected to OpenAI API")
        print(f"✓ Model {fallback_model} is working")
        print(f"✓ Response received: \"{response.choices[0].message.content}\"")
        print("===============================")
        
        # Return empty dict since we can't get actual rate limit info
        return {}
        
    except Exception as e:
        error_str = str(e).lower()
        is_quota_error = (
            'insufficient_quota' in error_str or 
            'code: 429' in error_str or 
            'exceeded your current quota' in error_str
        )
        
        if is_quota_error:
            print("\n=== API Quota Error ===")
            print("You've hit your API quota limit. This may indicate:")
            print("1. Your account doesn't have sufficient funds")
            print("2. You've exhausted your rate limits")
            print("3. Your account might be restricted")
            print("\nConsider checking your usage at: https://platform.openai.com/usage")
            print("And account billing at: https://platform.openai.com/account/billing")
            print("===============================")
        else:
            print(f"\nError connecting to API: {e}")
        
        print("\nProceeding without rate limit information.")
        
        check_anyway = input("\nWould you like to continue with the analysis anyway? (y/n): ")
        if check_anyway.lower() != 'y':
            print("Analysis cancelled.")
            return "ABORT"
            
        return {}

def create_interactive_html(results: Dict[str, Dict], policy: str, timestamp: str, 
                           safe_policy: str, model_name: str, samples_per_language: int) -> str:
    """
    Create an interactive HTML visualization using Plotly.js
    
    Args:
        results: Dictionary containing analysis results by language
        policy: The policy topic being analyzed
        timestamp: Timestamp string for the filename
        safe_policy: Sanitized policy name for the filename
        model_name: The name of the selected model
        samples_per_language: Number of samples per language
        
    Returns:
        Path to the saved HTML file
    """
    # Prepare data for visualization
    languages = []
    support_percentages = []
    oppose_percentages = []
    
    for language_name, data in results.items():
        total = data['support_count'] + data['oppose_count'] + data['error_count']
        if total > 0:
            languages.append(language_name)
            support_percent = (data['support_count'] / total) * 100
            oppose_percent = (data['oppose_count'] / total) * 100
            support_percentages.append(support_percent)
            oppose_percentages.append(oppose_percent)
    
    # Sort by support percentage (descending)
    sorted_data = sorted(zip(languages, support_percentages, oppose_percentages), 
                        key=lambda x: x[1], reverse=True)
    languages = [x[0] for x in sorted_data]
    support_percentages = [x[1] for x in sorted_data]
    oppose_percentages = [x[2] for x in sorted_data]
    
    # Calculate overall stats
    overall_support = sum(data['support_count'] for data in results.values())
    overall_total = sum(data['support_count'] + data['oppose_count'] + data['error_count'] for data in results.values())
    overall_rate = (overall_support / overall_total) * 100 if overall_total > 0 else 0
    
    # Find highest and lowest support
    highest_support = sorted_data[0] if sorted_data else (None, 0, 0)
    lowest_support = sorted_data[-1] if sorted_data else (None, 0, 0)
    
    # Calculate standard deviation if we have enough data
    if len(support_percentages) > 1:
        std_dev = (sum((x - sum(support_percentages) / len(support_percentages)) ** 2 
                   for x in support_percentages) / len(support_percentages)) ** 0.5
    else:
        std_dev = 0
    
    # Extract example arguments (if available)
    example_arguments = {}
    for language_name, data in results.items():
        example_arguments[language_name] = {
            "support": data['pros'][0] if data['pros'] and len(data['pros']) > 0 else "No example available.",
            "oppose": data['cons'][0] if data['cons'] and len(data['cons']) > 0 else "No example available."
        }
    
    # Convert policy to title case
    policy_title_case = ' '.join(word.capitalize() for word in policy.split())
    
    # Create HTML content with updated heading structure
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{policy_title_case} Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }}
        body {{
            background-color: #f8fafc;
            color: #334155;
            padding: 20px;
            min-height: 100vh;
        }}
        .dashboard-container {{
            max-width: 1200px;
            width: 95%;
            margin: 0 auto;
            background-color: white;
            border-radius: 16px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
            padding: 35px;
        }}
        .header {{
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            color: #4338ca;
            font-size: 28px;
            margin-bottom: 8px;
            font-weight: 700;
            background: linear-gradient(90deg, #4338ca, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header h2 {{
            color: #1e293b;
            font-size: 22px;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .header p {{
            color: #64748b;
            font-size: 16px;
            max-width: 700px;
            margin: 0 auto;
        }}
        .chart-container {{
            height: 550px;
            margin: 30px 0 30px 0;
            border-radius: 12px;
            overflow: visible;
            background-color: white;
            width: 100%;
        }}
        .stats-row {{
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            flex: 1;
            min-width: 180px;
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
            text-align: center;
            border-top: 4px solid #4338ca;
            transition: transform 0.2s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-card:nth-child(1) {{
            border-top-color: #6366f1;
        }}
        .stat-card:nth-child(2) {{
            border-top-color: #10b981;
        }}
        .stat-card:nth-child(3) {{
            border-top-color: #ef4444;
        }}
        .stat-card:nth-child(4) {{
            border-top-color: #f59e0b;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 8px;
        }}
        .stat-label {{
            color: #64748b;
            font-size: 14px;
            font-weight: 500;
        }}
        .info-section {{
            background-color: white;
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        }}
        .info-section h3 {{
            color: #4338ca;
            font-size: 18px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .info-section p {{
            color: #64748b;
            margin-bottom: 12px;
            line-height: 1.6;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #64748b;
            font-size: 14px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
        }}
        .method-specs {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }}
        .method-item {{
            flex: 1;
            min-width: 150px;
            background-color: #f8fafc;
            padding: 12px;
            border-radius: 8px;
        }}
        .method-item strong {{
            display: block;
            color: #4338ca;
            margin-bottom: 6px;
            font-size: 15px;
        }}
        .method-item span {{
            color: #64748b;
            font-size: 14px;
        }}
        @media (max-width: 768px) {{
            .stats-row {{
                flex-direction: column;
            }}
            .stat-card {{
                min-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <h1>APRIL: All Policy Really Is Local</h1>
            <h2>Support for {policy_title_case} by Language</h2>
            <p>Analysis of how {model_name} responds when prompted in different languages</p>
        </div>

        <div class="chart-container" id="chart-main"></div>
        
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{overall_rate:.1f}%</div>
                <div class="stat-label">Overall Support Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{highest_support[1]:.1f}%</div>
                <div class="stat-label">Highest Support ({highest_support[0]})</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{lowest_support[1]:.1f}%</div>
                <div class="stat-label">Lowest Support ({lowest_support[0]})</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{std_dev:.2f}</div>
                <div class="stat-label">Standard Deviation</div>
            </div>
        </div>

        <div class="info-section">
            <h3>Key Insights</h3>
            <p>This analysis reveals variation in how the AI model responds to {policy_title_case} questions when prompted in different languages.</p>
            <p>When prompted in {highest_support[0]}, the model expressed support at {highest_support[1]:.1f}%, while in {lowest_support[0]}, support was expressed at only {lowest_support[1]:.1f}%.</p>
            <p>The standard deviation of {std_dev:.2f} quantifies the dispersion in support rates across the tested languages, potentially highlighting linguistic and cultural factors influencing AI responses.</p>
            
            <h3 style="margin-top: 20px;">Methodology</h3>
            <div class="method-specs">
                <div class="method-item">
                    <strong>Model</strong>
                    <span>{model_name}</span>
                </div>
                <div class="method-item">
                    <strong>Date</strong>
                    <span>{datetime.now().strftime('%Y-%m-%d')}</span>
                </div>
                <div class="method-item">
                    <strong>Languages</strong>
                    <span>{len(languages)}</span>
                </div>
                <div class="method-item">
                    <strong>Samples per language</strong>
                    <span>{samples_per_language}</span>
                </div>
                <div class="method-item">
                    <strong>Total API calls</strong>
                    <span>{len(languages) * samples_per_language}</span>
                </div>
                <div class="method-item">
                    <strong>Analysis ID</strong>
                    <span>{timestamp}</span>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>APRIL: All Policy Really Is Local | Analysis Date: {datetime.now().strftime('%Y-%m-%d')}</p>
            <p>GitHub: <a href="https://github.com/economicalstories/april" style="color: #4338ca; text-decoration: none;">https://github.com/economicalstories/april</a></p>
        </div>
    </div>
"""

    # Update the script part to fix the legend positioning
    script_part = """
    <script>
        // Embedded data from the analysis
        const languageData = [
            {languages_and_data}
        ];
        
        // Example arguments for/against by language
        const exampleArguments = {
            {example_args_json}
        };

        // Extract languages and support values for charting
        const languages = languageData.map(d => d.language);
        const supportValues = languageData.map(d => d.support);
        const opposeValues = languageData.map(d => d.oppose);

        // Create the stacked bar chart visualization
        function createStackedBarChart() {
            // Prepare the hover templates with proper formatting
            const supportHoverTemplate = languages.map((lang, i) => {
                return `<b>${lang}: ${supportValues[i].toFixed(1)}% Support</b><br>${exampleArguments[lang].support}`;
            });
            
            const opposeHoverTemplate = languages.map((lang, i) => {
                return `<b>${lang}: ${opposeValues[i].toFixed(1)}% Oppose</b><br>${exampleArguments[lang].oppose}`;
            });
            
            const supportTrace = {
                x: languages,
                y: supportValues,
                type: 'bar',
                name: 'Support',
                marker: {
                    color: '#10b981',
                    opacity: 1.0,
                    line: {
                        color: '#10b981',
                        width: 0
                    }
                },
                text: supportValues.map(val => val.toFixed(1) + '%'),
                textposition: 'auto',
                textfont: {
                    color: 'white',
                    size: 14,
                    weight: 'bold'
                },
                hovertemplate: supportHoverTemplate,
                cliponaxis: false
            };

            const opposeTrace = {
                x: languages,
                y: opposeValues,
                type: 'bar',
                name: 'Oppose',
                marker: {
                    color: '#ef4444',
                    opacity: 1.0,
                    line: {
                        color: '#ef4444',
                        width: 0
                    }
                },
                text: opposeValues.map(val => val.toFixed(1) + '%'),
                textposition: 'auto',
                textfont: {
                    color: 'white',
                    size: 14,
                    weight: 'bold'
                },
                hovertemplate: opposeHoverTemplate,
                cliponaxis: false
            };

            const layout = {
                barmode: 'stack',
                bargap: 0.15,
                bargroupgap: 0.05,
                xaxis: {
                    title: {
                        text: '',
                        font: {
                            size: 14,
                            color: '#64748b'
                        }
                    },
                    tickangle: -30,
                    tickfont: {
                        color: '#64748b',
                        size: 13
                    },
                    automargin: true
                },
                yaxis: {
                    title: {
                        text: 'Percentage (%)',
                        font: {
                            size: 14,
                            color: '#64748b'
                        }
                    },
                    range: [0, 105],
                    tickfont: {
                        color: '#64748b'
                    },
                    gridcolor: '#f1f5f9'
                },
                margin: {
                    l: 60,
                    r: 30,
                    t: 5,
                    b: 70
                },
                legend: {
                    orientation: 'h',
                    y: -0.10,
                    x: 0.5,
                    xanchor: 'center',
                    yanchor: 'top',
                    font: {
                        family: 'Inter, sans-serif',
                        size: 14,
                        color: '#64748b'
                    },
                    bgcolor: 'rgba(255,255,255,0.9)',
                    bordercolor: '#e2e8f0',
                    borderwidth: 1
                },
                autosize: true,
                height: 520,
                width: null,
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                hoverlabel: {
                    bgcolor: 'white',
                    bordercolor: '#e2e8f0',
                    font: {
                        family: 'Inter, sans-serif',
                        size: 14,
                        color: '#1e293b'
                    },
                    align: 'left'
                },
                hovermode: 'closest'
            };

            const config = {
                responsive: true,
                displayModeBar: false,
                toImageButtonOptions: {
                    format: 'png',
                    filename: '{safe_policy}_support_by_language',
                    height: 800,
                    width: 1200,
                    scale: 2
                }
            };

            Plotly.newPlot('chart-main', [supportTrace, opposeTrace], layout, config);
            
            // Add resize listener to handle responsive resizing
            window.addEventListener('resize', function() {
                Plotly.Plots.resize(document.getElementById('chart-main'));
            });
        }

        // Initialize chart when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            createStackedBarChart();
        });
    </script>
</body>
</html>
    """
    
    # The replacements for policy_title_case and model_name aren't needed in the chart title anymore,
    # but we'll keep them for any other places they might be used
    script_part = script_part.replace("{policy_title_case}", policy_title_case)
    script_part = script_part.replace("{model_name}", model_name)
    script_part = script_part.replace("{safe_policy}", safe_policy)
    
    # Format the language data for JS
    languages_json = []
    for lang, support, oppose in zip(languages, support_percentages, oppose_percentages):
        languages_json.append(f'{{language: "{lang}", support: {support:.1f}, oppose: {oppose:.1f}}}')
    
    # Join the language JSON objects with commas
    languages_and_data = ",\n            ".join(languages_json)
    
    # Format the example arguments for JS
    example_args = []
    for lang in languages:
        if lang in example_arguments:
            support_arg = example_arguments[lang]["support"].replace('"', '\\"')
            oppose_arg = example_arguments[lang]["oppose"].replace('"', '\\"')
            example_args.append(f'"{lang}": {{\n                support: "{support_arg}",\n                oppose: "{oppose_arg}"\n            }}')
    
    # Join the example arguments JSON objects with commas
    example_args_json = ",\n            ".join(example_args)
    
    # Replace the placeholders with the formatted JSON
    script_part = script_part.replace("{languages_and_data}", languages_and_data)
    script_part = script_part.replace("{example_args_json}", example_args_json)
    
    # Combine HTML and script
    full_html = html_content + script_part
    
    # Save to file
    output_file = f"{safe_policy}_interactive_{timestamp}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return output_file

def main():
    """
    Main function to run the policy sentiment analysis
    """
    print("\n" + "="*50)
    print("APRIL: All Policy Really Is Local")
    print("Policy Sentiment Analysis Across Languages")
    print("="*50)
    
    # Check for API key
    if not OPENAI_AVAILABLE:
        print("Error: The OpenAI library is not installed. Please install it with 'pip install openai'.")
        return
    
    api_key = get_openai_api_key()
    if not api_key:
        print("No API key found. Please set your OpenAI API key.")
        return
    
    client = OpenAI(api_key=api_key)
    
    # Scan for existing analyses
    existing_analyses = scan_existing_analyses()
    
    # Present integrated menu options to the user
    print("\nOptions:")
    print("0. Run a new analysis")
    
    if existing_analyses:
        for i, analysis in enumerate(existing_analyses):
            lang_text = f"{analysis['languages']} languages"
            sample_text = f"{analysis['samples_per_language']} samples per language"
            print(f"{i+1}. Visualize existing analysis: {analysis['policy']} ({analysis['timestamp']}) - {lang_text}, {sample_text}")
    
    choice = -1
    max_choice = len(existing_analyses)
    while choice < 0 or choice > max_choice:
        try:
            choice = int(input("\nEnter your choice (0 for new analysis): "))
            if choice < 0 or choice > max_choice:
                print(f"Please enter a number between 0 and {max_choice}.")
        except ValueError:
            print("Please enter a valid number.")
    
    if choice > 0:
        # Visualize existing analysis
        selected_analysis = existing_analyses[choice-1]
        print(f"\nVisualizing existing analysis for {selected_analysis['policy']}...")
        visualize_existing_analysis(selected_analysis)
        return
    
    # Proceed with new analysis if chosen or no existing analyses found
    print("\nStarting new analysis...")
    
    # Step 1: Select model
    print("\nSelect model to use:")
    for key, model in MODEL_OPTIONS.items():
        print(f"{key}. {model['name']} - {model['description']}")
    
    model_choice = ""
    while model_choice not in MODEL_OPTIONS:
        model_choice = input("\nEnter model choice (default: 2 for GPT-4o-mini): ").strip()
        if model_choice == "":
            model_choice = "2"  # Default to GPT-4o-mini
    
    selected_model = MODEL_OPTIONS[model_choice]
    model_id = selected_model["id"]
    model_name = selected_model["name"]
    cost_per_call = selected_model["cost_per_call"]
    
    print(f"\nSelected model: {model_name}")
    
    # Check rate limits
    rate_info = check_rate_limits(client, model_id)
    if rate_info == "ABORT":
        return
    
    # Get the policy topic
    policy = input("Enter the policy topic to analyze (e.g., 'Universal Basic Income'): ")
    
    # Step 3: Show language options and get selection
    languages = [
        {"name": "English", "code": "English"},
        {"name": "Spanish", "code": "Spanish"},
        {"name": "French", "code": "French"},
        {"name": "German", "code": "German"},
        {"name": "Italian", "code": "Italian"},
        {"name": "Portuguese", "code": "Portuguese"},
        {"name": "Russian", "code": "Russian"},
        {"name": "Japanese", "code": "Japanese"},
        {"name": "Korean", "code": "Korean"},
        {"name": "Chinese", "code": "Chinese"},
        {"name": "Arabic", "code": "Arabic"},
        {"name": "Hindi", "code": "Hindi"}
    ]
    
    print("\nAvailable languages:")
    for i, lang in enumerate(languages):
        print(f"{i+1}. {lang['name']}")
    
    language_input = input("\nSelect languages (comma-separated numbers, or 'all'): ")
    
    selected_languages = []
    if language_input.lower() == 'all':
        selected_languages = languages
    else:
        try:
            lang_indices = [int(idx.strip()) - 1 for idx in language_input.split(',') if idx.strip()]
            for idx in lang_indices:
                if 0 <= idx < len(languages):
                    selected_languages.append(languages[idx])
                else:
                    print(f"Warning: {idx + 1} is not a valid language index, skipping.")
        except ValueError:
            print("Invalid input. Please use comma-separated numbers or 'all'.")
            return
    
    if not selected_languages:
        print("No languages selected. Exiting.")
        return
    
    print("\nSelected languages:")
    for lang in selected_languages:
        print(f"- {lang['name']}")
    
    # Step 3: Get the number of responses per language
    print("\nRecommended sample sizes:")
    print("- 1 sample per language (quick test)")
    print("- 10 samples per language (initial test for variance)")
    print("- 100 samples per language (good accuracy)")
    print("- 1000 samples per language (high accuracy)")
    
    samples_input = input("\nHow many responses would you like per language? (default: 10) ")
    
    try:
        # Use default value of 10 if input is empty
        if samples_input.strip() == "":
            samples = 10
        else:
            samples = int(samples_input)
        
        if samples <= 0:
            print("Number of samples must be positive. Exiting.")
            return
    except ValueError:
        print("Invalid input. Please enter a number. Exiting.")
        return
    
    # Calculate estimated API calls and cost
    total_api_calls = len(selected_languages) * samples
    estimated_cost = total_api_calls * cost_per_call
    estimated_tokens = total_api_calls * 1000  # Rough estimate of tokens per call
    
    print("\nAnalysis plan:")
    print(f"- Topic: {policy}")
    print(f"- Model: {model_name}")
    print(f"- Languages: {len(selected_languages)}")
    print(f"- Responses per language: {samples}")
    print(f"- Total API calls: {total_api_calls}")
    print(f"- Estimated cost: ${estimated_cost:.2f}")
    
    # Add rate limit check information
    if rate_info:
        requests_remaining = rate_info.get("requests_remaining", "Unknown")
        tokens_remaining = rate_info.get("tokens_remaining", "Unknown")
        
        if requests_remaining != "Unknown" and tokens_remaining != "Unknown":
            print(f"\nRate limit check:")
            if int(requests_remaining) < total_api_calls:
                print(f"⚠️ WARNING: Your planned analysis requires {total_api_calls} API calls, but you only have {requests_remaining} remaining!")
                print(f"   Consider reducing the sample size or waiting until limits reset in {rate_info.get('requests_reset', 'Unknown')}")
            else:
                print(f"✓ You have sufficient API call capacity ({requests_remaining} remaining, need {total_api_calls})")
            
            if int(tokens_remaining) < estimated_tokens:
                print(f"⚠️ WARNING: Your planned analysis may require ~{estimated_tokens} tokens, but you only have {tokens_remaining} remaining!")
                print(f"   Consider reducing the sample size or waiting until limits reset in {rate_info.get('tokens_reset', 'Unknown')}")
            else:
                print(f"✓ You have sufficient token capacity ({tokens_remaining} remaining, estimated need ~{estimated_tokens})")
    
    print("\nNote: Please check OpenAI's current pricing at https://openai.com/pricing before proceeding.")
    
    confirm = input("\nProceed with analysis? (y/n): ")
    if confirm.lower() != 'y':
        print("Analysis cancelled.")
        return
    
    # Generate timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output files
    output_csv = f"{policy.replace(' ', '_').lower()}_analysis_{timestamp}.csv"
    output_summary = f"{policy.replace(' ', '_').lower()}_summary_{timestamp}.txt"
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['language', 'sample_id', 'prompt', 'explanation', 'pro', 'con', 'support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Step 4: Run the tests
        results = {}
        
        total_iterations = len(selected_languages) * samples
        iteration = 0
        
        try:
            for language in selected_languages:
                language_name = language['name']
                language_code = language['code']
                
                print(f"\nProcessing {language_name}...")
                results[language_name] = {
                    'support_count': 0,
                    'oppose_count': 0,
                    'error_count': 0,
                    'explanations': [],
                    'pros': [],
                    'cons': [],
                    'prompts': []
                }
                
                for i in range(samples):
                    iteration += 1
                    current_progress = f"Progress: {iteration}/{total_iterations} - Running sample {i+1}/{samples} for {language_name}"
                    print(current_progress)
                    
                    while True:
                        try:
                            # Translate prompt if needed
                            prompt = translate_prompt(policy, language_code, client, model_id)
                            
                            # Store the prompt for reference
                            results[language_name]['prompts'].append(prompt)
                            
                            # Ask the policy question
                            response = ask_policy_question(prompt, client, language_code, model_id)
                            
                            # Process response
                            if response['support'] is not None:
                                if response['support'] == 1:
                                    results[language_name]['support_count'] += 1
                                else:
                                    results[language_name]['oppose_count'] += 1
                            else:
                                results[language_name]['error_count'] += 1
                            
                            results[language_name]['explanations'].append(response['explanation'])
                            results[language_name]['pros'].append(response['pro'])
                            results[language_name]['cons'].append(response['con'])
                            
                            # Write to CSV
                            writer.writerow({
                                'language': language_name,
                                'sample_id': i + 1,
                                'prompt': prompt,
                                'explanation': response['explanation'],
                                'pro': response['pro'],
                                'con': response['con'],
                                'support': response['support']
                            })
                            csvfile.flush()
                            
                            # Small delay to avoid rate limiting
                            time.sleep(0.5)
                            break  # Break the while loop on success
                        
                        # Special handling for AbortAnalysis - let it propagate up
                        except AbortAnalysis:
                            raise  # Re-raise to ensure it's caught at the top level
                            
                        except Exception as e:
                            action = handle_api_error(e, current_progress)
                            if action == "retry":
                                continue  # Try again after waiting
                            elif action == "abort":
                                print(f"\n{'='*60}")
                                print("ANALYSIS ABORTED BY USER")
                                print("No more API calls will be made. Proceeding with analysis of collected data.")
                                print(f"{'='*60}\n")
                                # Raise AbortAnalysis to break out of all loops
                                raise AbortAnalysis("User requested to abort the entire analysis")
                            else:
                                print(f"Error processing sample: {e}")
                                # Store as an error and continue to next sample
                                results[language_name]['error_count'] += 1
                                break  # Break the retry loop and move to next sample
                            
        except AbortAnalysis as e:
            print(f"\n{'='*60}")
            print(f"ANALYSIS ABORTED: {e}")
            print(f"No more API calls will be made. Proceeding with analysis of collected data.")
            print(f"{'='*60}\n")
            # Fall through to continue processing what we have
        
        # Generate summary and visualization with collected data
        print("\nGenerating summary statistics...")
        
        with open(output_summary, 'w', encoding='utf-8') as summary_file:
            summary_file.write(f"APRIL: All Policy Really Is Local\n")
            summary_file.write(f"Policy Analysis: {policy}\n")
            summary_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            summary_file.write(f"Total samples: {total_iterations}\n\n")
            
            summary_file.write("Results by Language:\n")
            summary_file.write("====================\n\n")
            
            for language_name, data in results.items():
                total = data['support_count'] + data['oppose_count'] + data['error_count']
                if total > 0:
                    support_percent = (data['support_count'] / total) * 100
                    oppose_percent = (data['oppose_count'] / total) * 100
                    error_percent = (data['error_count'] / total) * 100
                    
                    summary_file.write(f"{language_name}:\n")
                    summary_file.write(f"  Support: {data['support_count']} ({support_percent:.1f}%)\n")
                    summary_file.write(f"  Oppose: {data['oppose_count']} ({oppose_percent:.1f}%)\n")
                    if data['error_count'] > 0:
                        summary_file.write(f"  Errors: {data['error_count']} ({error_percent:.1f}%)\n")
                    summary_file.write("\n")
            
            summary_file.write("\nSummary:\n")
            summary_file.write("========\n\n")
            
            # Calculate overall statistics
            total_support = sum(data['support_count'] for data in results.values())
            total_oppose = sum(data['oppose_count'] for data in results.values())
            total_samples = total_support + total_oppose + sum(data['error_count'] for data in results.values())
            
            if total_samples > 0:
                overall_support_percent = (total_support / total_samples) * 100
                summary_file.write(f"Overall support across all languages: {overall_support_percent:.1f}%\n\n")
                
                # Find languages with highest and lowest support
                language_support = [(lang, (data['support_count'] / (data['support_count'] + data['oppose_count'] + data['error_count'])) * 100) 
                                    for lang, data in results.items() 
                                    if data['support_count'] + data['oppose_count'] + data['error_count'] > 0]
                
                if language_support:
                    most_supportive = max(language_support, key=lambda x: x[1])
                    least_supportive = min(language_support, key=lambda x: x[1])
                    
                    summary_file.write(f"Most supportive language: {most_supportive[0]} ({most_supportive[1]:.1f}%)\n")
                    summary_file.write(f"Least supportive language: {least_supportive[0]} ({least_supportive[1]:.1f}%)\n")
                    
                    # Calculate variance
                    if len(language_support) > 1:
                        support_values = [x[1] for x in language_support]
                        variance = sum((x - sum(support_values) / len(support_values)) ** 2 for x in support_values) / len(support_values)
                        summary_file.write(f"\nVariance in support across languages: {variance:.2f}\n")
                        summary_file.write(f"Standard deviation: {variance ** 0.5:.2f}\n")
            
            summary_file.write("\nAnalysis Details:\n")
            summary_file.write("=================\n\n")
            summary_file.write(f"Model: {model_name}\n")
            summary_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
            summary_file.write(f"Number of languages: {len(selected_languages)}\n")
            summary_file.write(f"Samples per language: {samples}\n")
            summary_file.write(f"Total API calls: {total_iterations}\n")
            summary_file.write(f"CSV file: {output_csv}\n")
            summary_file.write(f"Analysis ID: {timestamp}\n")
            summary_file.write(f"GitHub: https://github.com/economicalstories/april\n")
        
        # Create visualization if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            print("\nGenerating visualization...")
            viz_file = create_visualization(results, policy, timestamp, policy.replace(' ', '_').lower(), model_name)
            if viz_file:
                print(f"Visualization saved as: {viz_file}")
                
                # Optionally display the visualization (if in Jupyter notebook)
                if 'ipykernel' in sys.modules:
                    try:
                        from IPython.display import display, Image
                        display(Image(viz_file))
                    except ImportError:
                        pass
        else:
            print("\nVisualization skipped: matplotlib is not installed.")
        
        print(f"\nAnalysis complete!")
        print(f"Results saved to {output_csv} and {output_summary}")
        
        # Create interactive HTML visualization
        interactive_html = create_interactive_html(results, policy, timestamp, policy.replace(' ', '_').lower(), model_name, samples)
        if interactive_html:
            print(f"Interactive HTML visualization saved as: {interactive_html}")
        
if __name__ == "__main__":
    main() 