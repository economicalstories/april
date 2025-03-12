import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

class VisualizationService:
    def __init__(self, output_dir: str = 'data/outputs'):
        """Initialize the visualization service with output directories."""
        # Ensure we're not prepending 'backend' to the path
        if output_dir.startswith('backend/'):
            output_dir = output_dir[len('backend/'):]
        
        # Ensure we're using the correct top-level directory structure
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.analyses_dir = os.path.join(output_dir, 'analyses')
        self.summaries_dir = os.path.join(output_dir, 'summaries')

        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.analyses_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

    def create_visualization(self, results: Dict[str, Dict], policy: str, timestamp: str, model_name: str) -> str:
        """Create a visualization of the analysis results with a stacked bar chart."""
        # Set up figure style similar to the example image
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
        
        for lang in languages:
            support = results[lang]['support']
            oppose = results[lang]['oppose']
            total = support + oppose + results[lang]['error']
            
            support_counts.append(support)
            oppose_counts.append(oppose)
            total_counts.append(total)
            
            # Calculate percentages
            support_pct = (support / total * 100) if total > 0 else 0
            support_percentages.append(support_pct)
        
        # Combine data and sort by support percentage (descending)
        data = list(zip(languages, support_counts, oppose_counts, support_percentages))
        data.sort(key=lambda x: x[3], reverse=True)
        
        # Unpack sorted data
        sorted_languages = [item[0] for item in data]
        sorted_support_counts = [item[1] for item in data]
        sorted_oppose_counts = [item[2] for item in data]
        sorted_percentages = [item[3] for item in data]
        
        # Create horizontal bars (support only as in the reference image)
        y_pos = range(len(sorted_languages))
        bars = ax.barh(y_pos, sorted_percentages, height=0.6, 
                    color='#2ecc71', edgecolor='white', linewidth=0.5)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_languages, fontsize=14, fontweight='bold')
        ax.set_xlabel('Support Rate (%)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # Remove spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        # Add count and percentage labels on the bars
        for i, (bar, count, pct) in enumerate(zip(bars, sorted_support_counts, sorted_percentages)):
            if pct > 0:
                label = f"{count} ({pct:.1f}%)"
                text_color = 'white' if pct > 40 else 'black'
                # Position label in the middle of the bar
                ax.text(pct/2, bar.get_y() + bar.get_height()/2,
                        label, ha='center', va='center',
                        color=text_color, fontweight='bold', fontsize=12)
        
        # Add legend for support/oppose
        support_patch = mpatches.Patch(color='#2ecc71', label='Support')
        oppose_patch = mpatches.Patch(color='#e74c3c', label='Oppose')
        legend = ax.legend(handles=[support_patch, oppose_patch], 
                         loc='upper center', ncol=2,
                         bbox_to_anchor=(0.5, 1.05), fontsize=12)
        
        # Create title and subtitle
        # Format model name correctly (GPT-4o not GPT-4O)
        display_model = model_name
        if model_name.lower() == "gpt-4o":
            display_model = "gpt-4o"  # Ensure lowercase 'o'
        elif model_name.lower().startswith("gpt-"):
            # Capitalize but keep proper case for model variant
            model_parts = model_name.split('-')
            if len(model_parts) > 1:
                display_model = model_parts[0].upper() + '-' + model_parts[1]
                
        title = f"{policy} Support Analysis"
        subtitle = f"Model: {display_model}"
        ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
        fig.text(0.5, 0.92, subtitle, ha='center', fontsize=16)
        
        # Add summary footer
        total_samples = sum(support_counts) + sum(oppose_counts)
        samples_per_lang = total_samples // len(languages) if len(languages) > 0 else 0
        overall_support = sum(support_counts) / total_samples * 100 if total_samples > 0 else 0
        
        date_str = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        
        footer = (
            f"Analysis Summary   •   "
            f"Policy: {policy}   •   "
            f"Model: {display_model}   •   "
            f"Date: {date_str}   •   "
            f"Total Samples: {total_samples}   •   "
            f"Samples/Language: {samples_per_lang}   •   "
            f"Overall Support: {overall_support:.1f}%"
        )
        
        fig.text(0.5, 0.01, footer, ha='center', va='bottom', fontsize=11,
                bbox=dict(facecolor='whitesmoke', edgecolor='lightgray', boxstyle='round,pad=0.5'),
                fontweight='normal', style='italic')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the visualization
        output_file = os.path.join(
            self.viz_dir,
            f"{policy.lower().replace(' ', '_')}_visualization_{timestamp}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the relative path (without 'backend/')
        if output_file.startswith('backend/'):
            output_file = output_file[len('backend/'):]
            
        return output_file

    def create_summary(self, results: Dict[str, Dict], policy: str, timestamp: str, 
                      model_name: str, samples_per_language: int) -> str:
        """Create a summary text file of the analysis results."""
        summary_lines = [
            f"Policy Analysis Summary",
            f"Policy: {policy}",
            f"Model: {model_name}",
            f"Timestamp: {timestamp}",
            f"Samples per language: {samples_per_language}",
            "",
            "Results by Language:",
            "-" * 40
        ]

        overall_support = 0
        total_samples = 0
        support_rates = []

        for lang in results:
            total = results[lang]['support'] + results[lang]['oppose'] + results[lang]['error']
            if total > 0:
                support_rate = (results[lang]['support'] / total) * 100
                support_rates.append(support_rate)
                overall_support += results[lang]['support']
                total_samples += total
                
                summary_lines.extend([
                    f"\n{lang}:",
                    f"Support: {results[lang]['support']} ({support_rate:.1f}%)",
                    f"Oppose: {results[lang]['oppose']}",
                    f"Error: {results[lang]['error']}"
                ])

        if support_rates:
            avg_support = sum(support_rates) / len(support_rates)
            std_dev = (sum((x - avg_support) ** 2 for x in support_rates) / len(support_rates)) ** 0.5
            
            summary_lines.extend([
                "\nOverall Statistics:",
                "-" * 40,
                f"Average support rate: {avg_support:.1f}%",
                f"Standard deviation: {std_dev:.2f}",
                f"Total samples analyzed: {total_samples}"
            ])

        output_file = os.path.join(
            self.summaries_dir,
            f"{policy.lower().replace(' ', '_')}_summary_{timestamp}.txt"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        # Return the relative path (without 'backend/')
        if output_file.startswith('backend/'):
            output_file = output_file[len('backend/'):]
            
        return output_file 