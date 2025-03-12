import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import pandas as pd  # Add pandas import
import re  # For text cleaning
import csv  # For CSV quoting

# Add the backend directory to the path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, backend_dir)

# Import the required modules
from src.services.openai_service import OpenAIService

def get_user_input(prompt, default=None):
    """Get user input with a default value."""
    if default is not None:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def get_languages_input():
    """Get languages input from user with support for numeric selection and 'all' option."""
    # Define all available languages
    all_languages = [
        "English", "Spanish", "French", "German", "Italian", "Portuguese",
        "Russian", "Japanese", "Korean", "Chinese", "Arabic", "Hindi"
    ]
    
    print("\nLanguage Selection:")
    print("------------------")
    print("Options:")
    print("  'all' - Select all languages")
    print("  Numbers (comma-separated) - Select by number")
    print("  Names (comma-separated) - Select by name\n")
    
    # Display numbered list of languages
    for i, lang in enumerate(all_languages, 1):
        print(f"  {i}. {lang}")
    
    user_input = input("\nEnter your selection: ").strip().lower()
    
    # Handle 'all' option
    if user_input == 'all':
        return all_languages
    
    # Try to parse as numbers
    try:
        # Check if input contains numbers
        if all(part.strip().isdigit() for part in user_input.split(',')):
            selected_indices = [int(x.strip()) for x in user_input.split(',')]
            selected_languages = []
            for idx in selected_indices:
                if 1 <= idx <= len(all_languages):
                    selected_languages.append(all_languages[idx-1])
                else:
                    print(f"Warning: Ignoring invalid number {idx}")
            if selected_languages:
                return selected_languages
            print("No valid numbers provided. Please try again.")
            return get_languages_input()
    except ValueError:
        pass
    
    # Handle language names
    selected_languages = [lang.strip() for lang in user_input.split(',')]
    valid_languages = []
    for lang in selected_languages:
        lang_title = lang.title()
        if lang_title in all_languages:
            valid_languages.append(lang_title)
        else:
            print(f"Warning: Ignoring invalid language '{lang}'")
    
    if valid_languages:
        return valid_languages
    
    print("No valid languages provided. Please try again.")
    return get_languages_input()

def clean_text_for_csv(text):
    """Clean text by replacing newlines and carriage returns with spaces for CSV output.
    Also handles commas and quotes to ensure proper CSV formatting."""
    if text is None:
        return ""
    
    # Convert to string in case it's not already
    text = str(text)
    
    # Replace any combination of newlines and carriage returns with a single space
    cleaned = re.sub(r'[\r\n]+', ' ', text)
    
    # Replace multiple spaces with a single space
    cleaned = re.sub(r' +', ' ', cleaned)
    
    # Remove any trailing/leading whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def run_analysis():
    """Run analysis with user-specified parameters."""
    
    # Default parameters
    default_policy = "Nationalise Industry"
    default_samples = 10
    default_model = "gpt-4o"
    
    # Get parameters from user
    print("\n=== APRIL Analysis Configuration ===")
    policy = get_user_input("Enter policy to analyze", default_policy)
    languages = get_languages_input()
    samples_per_language = int(get_user_input("Enter samples per language", default_samples))
    model = get_user_input("Enter model name", default_model)
    
    # Confirm configuration
    print("\nConfiguration Summary:")
    print(f"Policy: {policy}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Samples per language: {samples_per_language}")
    print(f"Model: {model}")
    print(f"Total API calls: {len(languages) * samples_per_language}")
    
    confirm = get_user_input("\nProceed with analysis? (y/n)", "y")
    if confirm.lower() != "y":
        print("Analysis cancelled.")
        return
    
    print("\nStarting analysis...")
    
    # Initialize the OpenAI service
    openai_service = OpenAIService()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Initialize results dictionary
    results = {lang: {"support": 0, "oppose": 0, "error": 0, "pros": [], "cons": []} for lang in languages}
    
    # Initialize list to store all responses for CSV
    all_responses = []
    
    # Run analysis for each language and sample
    for lang in languages:
        print(f"\nAnalyzing policy in {lang}...")
        for i in range(samples_per_language):
            print(f"  Sample {i+1}/{samples_per_language}...", end="", flush=True)
            try:
                response = openai_service.analyze_policy(policy, lang, model)
                
                # Update results
                if response["support"] == 1:
                    results[lang]["support"] += 1
                    print(" Support", flush=True)
                else:
                    results[lang]["oppose"] += 1
                    print(" Oppose", flush=True)
                
                # Store unique pros and cons
                if response.get("pro") and response["pro"] not in results[lang]["pros"]:
                    results[lang]["pros"].append(response["pro"])
                if response.get("con") and response["con"] not in results[lang]["cons"]:
                    results[lang]["cons"].append(response["con"])
                
                # Store response for CSV (only essential fields)
                all_responses.append({
                    "language": lang,
                    "sample": i + 1,
                    "prompt": clean_text_for_csv(response.get("user_prompt", "")),
                    "support": response["support"],
                    "explanation": clean_text_for_csv(response.get("explanation", "")),
                    "pro": clean_text_for_csv(response.get("pro", "")),
                    "con": clean_text_for_csv(response.get("con", "")),
                    "raw_response": clean_text_for_csv(response.get("raw_response", ""))
                })
                        
            except Exception as e:
                print(f" Error: {str(e)}")
                results[lang]["error"] += 1
                
                # Store error in responses (only essential fields)
                all_responses.append({
                    "language": lang,
                    "sample": i + 1,
                    "prompt": "",
                    "support": None,
                    "explanation": f"Error: {clean_text_for_csv(str(e))}",
                    "pro": "",
                    "con": "",
                    "raw_response": ""
                })
    
    # Create and save CSV file
    output_dir = "data/outputs"
    analyses_dir = os.path.join(output_dir, "analyses")
    os.makedirs(analyses_dir, exist_ok=True)
    
    csv_file = os.path.join(
        analyses_dir,
        f"{policy.lower().replace(' ', '_')}_analysis_{timestamp}.csv"
    )
    
    # Add timestamp to dataframe
    df = pd.DataFrame(all_responses)
    
    # Reorder columns to put prompt near the beginning
    columns_order = [
        "language", "sample", "prompt", "support", "explanation", 
        "pro", "con", "timestamp", "original_policy", "model",
        "raw_response"
    ]
    
    # Add timestamp and other fields
    df['timestamp'] = timestamp
    df['original_policy'] = policy
    df['model'] = model
    
    # Reorder columns (will only include columns that exist)
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]
    
    # Remove redundant columns
    if 'system_prompt' in df.columns:
        df = df.drop('system_prompt', axis=1)
    if 'user_prompt' in df.columns:
        df = df.drop('user_prompt', axis=1)

    # Save as CSV
    df.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"\nAnalysis CSV saved: {csv_file}")
    
    # Create visualization
    create_visualization(results, policy, timestamp, model)
    
    # Create summary
    create_summary(results, policy, timestamp, model, samples_per_language)
    
    print("\nAnalysis completed!")
    print("Check data/outputs/visualizations and data/outputs/summaries for results")

def create_visualization(results, policy, timestamp, model_name):
    """Create a visualization of the analysis results with a stacked bar chart."""
    output_dir = "data/outputs"
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set up figure style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': [16, 12],
        'font.size': 12,
        'font.family': 'sans-serif',
    })
    
    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Extract data
    languages = list(results.keys())
    support_counts = []
    oppose_counts = []
    total_counts = []
    support_percentages = []
    oppose_percentages = []
    
    for lang in languages:
        support = results[lang]['support']
        oppose = results[lang]['oppose']
        total = support + oppose + results[lang]['error']
        
        support_counts.append(support)
        oppose_counts.append(oppose)
        total_counts.append(total)
        
        # Calculate percentages
        support_pct = (support / total * 100) if total > 0 else 0
        oppose_pct = (oppose / total * 100) if total > 0 else 0
        support_percentages.append(support_pct)
        oppose_percentages.append(oppose_pct)
    
    # Combine data and sort by support percentage (descending)
    data = list(zip(languages, support_counts, oppose_counts, support_percentages, oppose_percentages))
    data.sort(key=lambda x: x[3], reverse=True)
    
    # Unpack sorted data
    sorted_languages = [item[0] for item in data]
    sorted_support_counts = [item[1] for item in data]
    sorted_oppose_counts = [item[2] for item in data]
    sorted_support_percentages = [item[3] for item in data]
    sorted_oppose_percentages = [item[4] for item in data]
    
    # Create stacked horizontal bars
    y_pos = range(len(sorted_languages))
    
    # Support bars (first segment)
    support_bars = ax.barh(y_pos, sorted_support_percentages, height=0.6, 
                color='#2ecc71', edgecolor='white', linewidth=0.5, label='Support')
    
    # Oppose bars (second segment, stacked on top of support)
    oppose_bars = ax.barh(y_pos, sorted_oppose_percentages, height=0.6, 
                 color='#e74c3c', edgecolor='white', linewidth=0.5, left=sorted_support_percentages, label='Oppose')
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_languages, fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    # Add count and percentage labels on the support bars
    for i, (bar, count, pct) in enumerate(zip(support_bars, sorted_support_counts, sorted_support_percentages)):
        if pct > 10:  # Only show label if there's enough space
            label = f"{count} ({pct:.1f}%)"
            text_color = 'white' if pct > 40 else 'black'
            # Position label in the middle of the bar
            ax.text(pct/2, bar.get_y() + bar.get_height()/2,
                    label, ha='center', va='center',
                    color=text_color, fontweight='bold', fontsize=12)
    
    # Add count and percentage labels on the oppose bars
    for i, (bar, count, pct, support_pct) in enumerate(zip(oppose_bars, sorted_oppose_counts, sorted_oppose_percentages, sorted_support_percentages)):
        if pct > 10:  # Only show label if there's enough space
            label = f"{count} ({pct:.1f}%)"
            text_color = 'white' if pct > 40 else 'black'
            # Position label in the middle of the bar
            ax.text(support_pct + pct/2, bar.get_y() + bar.get_height()/2,
                    label, ha='center', va='center',
                    color=text_color, fontweight='bold', fontsize=12)
    
    # Add legend below the chart
    legend = ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.08),  # Position below the chart
        ncol=2,                      # Two columns (Support, Oppose)
        fontsize=12,
        frameon=True,               # Add a frame around the legend
        fancybox=True,              # Rounded corners
        shadow=True                 # Add shadow
    )
    
    # Create title and subtitle with proper formatting
    title = f"{policy} Support Analysis"
    subtitle = f"Model: {model_name}"
    
    # Position title and subtitle with proper spacing
    fig.text(0.5, 0.95, title, ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.91, subtitle, ha='center', fontsize=16)
    
    # Remove the ax.set_title() call since we're using fig.text()
    ax.set_title("")  # Clear any existing title
    
    # Add summary footer
    total_samples = sum(support_counts) + sum(oppose_counts)
    samples_per_lang = total_samples // len(languages) if len(languages) > 0 else 0
    overall_support = sum(support_counts) / total_samples * 100 if total_samples > 0 else 0
    
    date_str = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    
    footer = (
        f"Analysis Summary   •   "
        f"Policy: {policy}   •   "
        f"Model: {model_name}   •   "
        f"Date: {date_str}   •   "
        f"Total Samples: {total_samples}   •   "
        f"Samples/Language: {samples_per_lang}   •   "
        f"Overall Support: {overall_support:.1f}%"
    )
    
    fig.text(0.5, 0.01, footer, ha='center', va='bottom', fontsize=11,
            bbox=dict(facecolor='whitesmoke', edgecolor='lightgray', boxstyle='round,pad=0.5'),
            fontweight='normal', style='italic')
    
    # Adjust layout to accommodate the title, subtitle, and legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.88])
    
    # Save the visualization
    output_file = os.path.join(
        viz_dir,
        f"{policy.lower().replace(' ', '_')}_visualization_{timestamp}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved: {output_file}")
    return output_file

def create_summary(results, policy, timestamp, model_name, samples_per_language):
    """Create a summary text file of the analysis results."""
    output_dir = "data/outputs"
    summaries_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)
    
    summary_lines = [
        f"APRIL: All Policy Really Is Local",
        f"Policy Analysis: {policy}",
        f"Date: {timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}",
        f"Total samples: {len(results.keys()) * samples_per_language}",
        "",
        "Results by Language:",
        "====================",
        ""
    ]

    overall_support = 0
    total_samples = 0
    support_rates = []
    language_results = []

    for lang in results:
        total = results[lang]['support'] + results[lang]['oppose'] + results[lang]['error']
        if total > 0:
            support_rate = (results[lang]['support'] / total) * 100
            support_rates.append(support_rate)
            overall_support += results[lang]['support']
            total_samples += total
            
            language_results.append((lang, support_rate))
            
            summary_lines.extend([
                f"{lang}: {policy}",
                f"  Support: {results[lang]['support']} ({support_rate:.1f}%)",
                f"  Oppose: {results[lang]['oppose']} ({100-support_rate:.1f}%)",
                ""
            ])

    # Sort languages by support rate
    language_results.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate statistics
    avg_support = sum(support_rates) / len(support_rates) if support_rates else 0
    variance = sum((x - avg_support) ** 2 for x in support_rates) / len(support_rates) if support_rates else 0
    std_dev = variance ** 0.5
    
    summary_lines.extend([
        "",
        "Summary:",
        "========",
        "",
        f"Overall support across all languages: {avg_support:.1f}%",
        "",
        f"Most supportive language: {language_results[0][0]} ({language_results[0][1]:.1f}%)" if language_results else "",
        f"Least supportive language: {language_results[-1][0]} ({language_results[-1][1]:.1f}%)" if language_results else "",
        "",
        f"Variance in support across languages: {variance:.2f}",
        f"Standard deviation: {std_dev:.2f}",
        "",
        "Analysis Details:",
        "=================",
        "",
        f"Model: {model_name}",
        f"Date: {timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}",
        f"Number of languages: {len(results.keys())}",
        f"Samples per language: {samples_per_language}",
        f"Total API calls: {len(results.keys()) * samples_per_language}",
        f"Analysis ID: {timestamp}",
        "GitHub: https://github.com/economicalstories/april"
    ])

    output_file = os.path.join(
        summaries_dir,
        f"{policy.lower().replace(' ', '_')}_summary_{timestamp}.txt"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Summary saved: {output_file}")
    return output_file

if __name__ == "__main__":
    run_analysis() 