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

def translate_prompt(policy: str, language: str, client: Any) -> str:
    """
    Translate the policy question prompt into the specified language.
    
    Args:
        policy: The policy topic to ask about
        language: Target language for translation
        client: OpenAI client object
        
    Returns:
        Translated prompt string
    """
    if language.lower() == "english":
        # No need to translate
        return f"Explain in a sentence what {policy} is, and then indicate if you support this policy with 1 for yes or 0 for no."
    
    system_message = (
        "You are a professional translator. Translate the following English text "
        f"accurately into {language}, maintaining the same meaning and formality."
    )
    
    english_prompt = f"Explain in a sentence what {policy} is, and then indicate if you support this policy with 1 for yes or 0 for no."
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": english_prompt}
            ],
            temperature=0.1  # Low temperature for more consistent translation
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        else:
            print(f"Error translating to {language}. Using English prompt.")
            return english_prompt
    except Exception as e:
        print(f"Translation error: {e}")
        return english_prompt

def ask_policy_question(prompt: str, client: Any, language_code: str) -> Dict[str, Any]:
    """
    Ask a policy question to ChatGPT and get the response.
    
    Args:
        prompt: The policy question prompt
        client: OpenAI client object
        language_code: The language code to ensure response in the same language
        
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,  # Higher temperature for more opinion variation
            response_format={"type": "json_object"}
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            try:
                # Parse JSON response
                result = json.loads(content)
                # Ensure we have all required fields
                required_fields = ["explanation", "pro", "con", "support"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    raise ValueError(f"Response missing required fields: {', '.join(missing_fields)}")
                
                # Convert support to integer if it's not already
                if not isinstance(result["support"], int):
                    # Try to convert string to int
                    result["support"] = int(result["support"])
                
                return result
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing response: {e}")
                print(f"Raw response: {content}")
                # Return default values
                return {
                    "explanation": "Error parsing response",
                    "pro": "Error parsing response",
                    "con": "Error parsing response",
                    "support": None
                }
        else:
            print("Empty or invalid response from API")
            return {
                "explanation": "No response from API",
                "pro": "No response from API",
                "con": "No response from API",
                "support": None
            }
    except Exception as e:
        print(f"API request error: {e}")
        return {
            "explanation": f"API error: {str(e)}",
            "pro": f"API error: {str(e)}",
            "con": f"API error: {str(e)}",
            "support": None
        }

def create_visualization(results: Dict[str, Dict], policy: str, timestamp: str, safe_policy: str) -> str:
    """
    Create a bar chart visualization of the policy sentiment analysis results.
    
    Args:
        results: Dictionary containing analysis results by language
        policy: The policy topic being analyzed
        timestamp: Timestamp string for the filename
        safe_policy: Sanitized policy name for the filename
        
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
        f"Model: GPT-4o\n"
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
    Scan for existing analysis files (CSV and summary text files)
    
    Returns:
        List of dictionaries containing information about existing analyses
    """
    analyses = []
    
    # Pattern to match analysis CSV files
    pattern = r'(.+)_analysis_(\d{8}_\d{6})\.csv'
    
    # Look for CSV files in the current directory
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and '_analysis_' in f]
    
    for file in csv_files:
        match = re.match(pattern, file)
        if match:
            policy_name = match.group(1).replace('_', ' ')
            timestamp = match.group(2)
            
            # Check if there's a corresponding summary file
            summary_file = f"{policy_name.replace(' ', '_')}_summary_{timestamp}.txt"
            has_summary = os.path.exists(summary_file)
            
            # Check if there's a corresponding visualization file
            viz_file = f"{policy_name.replace(' ', '_')}_visualization_{timestamp}.png"
            has_viz = os.path.exists(viz_file)
            
            # Get number of languages and samples by scanning the CSV
            languages = set()
            samples = 0
            
            try:
                with open(file, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        languages.add(row['language'])
                        samples += 1
            except Exception as e:
                print(f"Error reading {file}: {e}")
            
            analyses.append({
                'policy': policy_name,
                'timestamp': timestamp,
                'csv_file': file,
                'summary_file': summary_file if has_summary else None,
                'viz_file': viz_file if has_viz else None,
                'languages': len(languages),
                'samples': samples,
                'samples_per_language': samples // len(languages) if len(languages) > 0 else 0
            })
    
    # Sort by timestamp (newest first)
    analyses.sort(key=lambda x: x['timestamp'], reverse=True)
    
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
    safe_policy = policy.replace(' ', '_')
    
    # Load results from CSV
    results = load_analysis_results(csv_file)
    
    # Create visualization
    viz_file = create_visualization(results, policy, timestamp, safe_policy)
    
    if viz_file:
        print(f"\nVisualization created: {viz_file}")
        
        # Optionally display the visualization (if in Jupyter notebook)
        if 'ipykernel' in sys.modules:
            try:
                from IPython.display import display, Image
                display(Image(viz_file))
            except ImportError:
                pass

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
    
    # Present options to the user
    if existing_analyses:
        print("\nExisting analyses found:")
        for i, analysis in enumerate(existing_analyses):
            print(f"{i+1}. {analysis['policy']} ({analysis['timestamp']}) - {analysis['languages']} languages, {analysis['samples_per_language']} samples per language")
        
        print("\nOptions:")
        print("0. Run a new analysis")
        for i, analysis in enumerate(existing_analyses):
            print(f"{i+1}. Visualize existing analysis: {analysis['policy']}")
        
        choice = -1
        while choice < 0 or choice > len(existing_analyses):
            try:
                choice = int(input("\nEnter your choice (0 for new analysis): "))
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
    
    # Step 1: Get the policy topic
    policy = input("Enter the policy topic to analyze (e.g., 'Universal Basic Income'): ")
    
    # Step 2: Create a safe version of the policy name for filenames
    safe_policy = re.sub(r'[^\w\s]', '', policy).replace(' ', '_').lower()
    
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
    print("- 1 sample per language (quick test) - 1 API call per language")
    print("- 10 samples per language (moderate accuracy) - 10 API calls per language")
    print("- 100 samples per language (good accuracy) - 100 API calls per language")
    print("- 1000 samples per language (high accuracy) - 1000 API calls per language")
    print("\nNote: Please check OpenAI's current pricing at https://openai.com/pricing before running large analyses.")
    
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
    estimated_cost = total_api_calls * 0.004  # Estimated cost based on GPT-4o as of March 2023
    
    print("\nAnalysis plan:")
    print(f"- Topic: {policy}")
    print(f"- Languages: {len(selected_languages)}")
    print(f"- Responses per language: {samples}")
    print(f"- Total API calls: {total_api_calls}")
    print(f"- Estimated cost: ${estimated_cost:.2f}")
    
    confirm = input("\nProceed with analysis? (y/n): ")
    if confirm.lower() != 'y':
        print("Analysis cancelled.")
        return
    
    # Generate timestamp for the output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output files
    output_csv = f"{safe_policy}_analysis_{timestamp}.csv"
    output_summary = f"{safe_policy}_summary_{timestamp}.txt"
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['language', 'sample_id', 'prompt', 'explanation', 'pro', 'con', 'support']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Step 4: Run the tests
        results = {}
        
        total_iterations = len(selected_languages) * samples
        iteration = 0
        
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
                print(f"Progress: {iteration}/{total_iterations} - Running sample {i+1}/{samples} for {language_name}")
                
                # Translate prompt if needed
                prompt = translate_prompt(policy, language_code, client)
                
                # Store the prompt for reference
                results[language_name]['prompts'].append(prompt)
                
                # Ask the policy question
                response = ask_policy_question(prompt, client, language_code)
                
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
                
                # Update results
                if response['support'] == 1:
                    results[language_name]['support_count'] += 1
                elif response['support'] == 0:
                    results[language_name]['oppose_count'] += 1
                else:
                    results[language_name]['error_count'] += 1
                
                results[language_name]['explanations'].append(response['explanation'])
                results[language_name]['pros'].append(response['pro'])
                results[language_name]['cons'].append(response['con'])
                
                # Save after each sample in case of interruption
                csvfile.flush()
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
        
        # Step 5: Generate summary statistics
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
            summary_file.write(f"Model: GPT-4o\n")
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
            viz_file = create_visualization(results, policy, timestamp, safe_policy)
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
        
if __name__ == "__main__":
    # Add import for Jupyter notebook support
    import sys
    main() 