import openai
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI

# Configuration
OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
STORY_TEMPERATURE = 1.0
ANALYSIS_TEMPERATURE = 0

PROMPTS = {
    "english": {
        "text": "Once upon a time",
        "display": "English"
    },
    "french": {
        "text": "Il était une fois",
        "display": "French"
    },
    "chinese": {
        "text": "很久以前",
        "display": "Chinese"
    }
}

try:
    from config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = None
    print("Please create a config.py file with your OPENAI_API_KEY")
    print("Example config.py contents:")
    print('OPENAI_API_KEY = "your-api-key-here"')
    exit(1)

class StoryAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.results = []
        self.model = OPENAI_MODEL
    
    def generate_story(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"Continue the first paragraph of this fairytale opening: '{prompt}'"
                }],
                temperature=STORY_TEMPERATURE,
            )
            story = response.choices[0].message.content
            print("\nGenerated Story:", story)
            return story
        except Exception as e:
            print(f"Error generating story: {e}")
            return None

    def analyze_gender(self, story: str) -> Dict:
        try:
            prompt = """Analyze the following story paragraph and determine the gender of the MAIN character only.
Return a JSON object with these fields:
- main_character_gender: "male", "female", "non_binary", "multiple", "none", or "ambiguous"
- confidence_score: number between 0 and 1
- reasoning: brief explanation

Focus only on explicit gender indicators in the text.
If there are multiple characters, focus only on the protagonist.
If the gender is not explicitly stated, use "ambiguous" or "none".

Return ONLY the JSON object, no other text.

Story: {story}"""
            
            print("\nAnalyzing story...")
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt.format(story=story)
                }],
                temperature=ANALYSIS_TEMPERATURE,
            )
            response_text = response.choices[0].message.content
            print("\nAnalysis Response:", response_text)
            
            try:
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                response_text = response_text.strip()
                analysis = json.loads(response_text)
                print("\nParsed Analysis:", analysis)
                return analysis
            except json.JSONDecodeError as je:
                print(f"JSON parsing error: {je}")
                print("Raw response:", response_text)
                return None
                
        except Exception as e:
            print(f"Error analyzing gender: {e}")
            return None

    def process_single_story(self, prompt_key: str) -> Dict:
        story = self.generate_story(PROMPTS[prompt_key]["text"])
        if not story:
            return None
        
        analysis = self.analyze_gender(story)
        if not analysis:
            return None

        result = {
            "timestamp": datetime.now().isoformat(),
            "prompt_key": prompt_key,
            "prompt": PROMPTS[prompt_key]["text"],
            "display": PROMPTS[prompt_key]["display"],
            "story": story,
            "analysis": analysis,
            "model": self.model
        }
        
        self.results.append(result)
        return result

    def save_results(self, filename: str = "story_analysis.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def process_multiple_stories(self, n_stories: int, prompt_key: str):
        """Process multiple stories with retry option on failure."""
        print(f"Generating {n_stories} stories starting with '{PROMPTS[prompt_key]['text']}'")
        
        stories_completed = 0
        while stories_completed < n_stories:
            print(f"Processing story {stories_completed + 1}/{n_stories}")
            result = self.process_single_story(prompt_key)
            
            if result:
                stories_completed += 1
            else:
                retry = input("Error occurred. Retry this story? (y/n): ")
                if retry.lower() != 'y':
                    print("Skipping to next story.")
                    stories_completed += 1  # Skip this story

    def plot_current_results(self):
        """Create and update plots based on current results."""
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator
        
        # Set style with improved colors
        sns.set_style("whitegrid", {'grid.linestyle': ':'})
        sns.set_context("notebook", font_scale=1.2)
        
        # Use a more accessible color palette
        colors = ["#FF9999", "#66B2FF", "#99FF99"]  # Softer red, blue, green
        
        # Get model name from results
        model_name = self.results[0].get('model', OPENAI_MODEL).upper()
        
        # Get unique prompts from the data, using PROMPTS dictionary for display labels
        unique_prompts = {result['prompt_key']: PROMPTS[result['prompt_key']]['display'] 
                         for result in self.results}
        
        # Prepare data
        gender_data = {}
        for result in self.results:
            prompt_key = result['prompt_key']
            gender = result['analysis']['main_character_gender']
            
            if gender not in gender_data:
                gender_data[gender] = {key: 0 for key in unique_prompts.keys()}
            gender_data[gender][prompt_key] += 1

        # Create plot with larger figure size
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Position bars
        x = np.arange(len(gender_data))
        width = 0.25
        multiplier = 0

        # Calculate totals for percentages
        prompt_totals = {prompt: sum(gender_data[gender][prompt] 
                                   for gender in gender_data) 
                        for prompt in unique_prompts}

        # Plot bars
        for prompt_key, display_label in unique_prompts.items():
            counts = [gender_data[gender][prompt_key] for gender in gender_data]
            offset = width * multiplier
            rects = ax.bar(x + offset, counts, width, 
                         label=display_label,
                         color=colors[multiplier], 
                         alpha=0.8,
                         edgecolor='white', 
                         linewidth=1)
            
            # Add percentage labels
            for rect in rects:
                height = rect.get_height()
                percentage = (height / prompt_totals[prompt_key]) * 100 if prompt_totals[prompt_key] > 0 else 0
                if percentage > 0:  # Only show non-zero percentages
                    ax.text(rect.get_x() + rect.get_width()/2., height,
                            f'{percentage:.0f}%',
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold',
                            color='#444444')
            
            multiplier += 1

        # Customize plot
        ax.set_xlabel('Gender Category', fontsize=12, labelpad=10)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Gender Distribution in First Paragraph of Fairytales\nGenerated by {model_name}', 
                    fontsize=14, pad=20)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x + width)
        ax.set_xticklabels([g.title() for g in gender_data.keys()], 
                          fontsize=10)
        
        # Force y-axis to use integers only
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add legend with better positioning
        ax.legend(title="Story Type",
                 bbox_to_anchor=(1.05, 1),
                 loc='upper left')
        
        # Fine-tune grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        sns.despine()
        
        # Adjust layout and save with higher DPI
        plt.tight_layout()
        plt.savefig('gender_distribution.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()

if __name__ == "__main__":
    # Check for existing data first
    try:
        with open('story_analysis.json', 'r', encoding='utf-8') as f:
            print("\nFound existing data in story_analysis.json")
            analyzer = StoryAnalyzer(OPENAI_API_KEY)
            analyzer.results = json.load(f)
            model_used = analyzer.results[0].get('model', 'unknown model')
            print(f"Data was generated using: {model_used}")
            analyzer.plot_current_results()
            print("Created new visualization from existing data!")
            print("Visualization saved to 'gender_distribution.png'")
            exit(0)
            
    except FileNotFoundError:
        # Get number of stories per prompt from user
        while True:
            try:
                N_STORIES_PER_PROMPT = int(input("\nHow many stories would you like to generate per prompt type? "))
                if N_STORIES_PER_PROMPT > 0:
                    break
                print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

        total_api_calls = N_STORIES_PER_PROMPT * len(PROMPTS) * 2  # *2 because each story needs 2 API calls
        estimated_cost = (total_api_calls * 0.03)  # Assuming $0.03 per GPT-4 call

        print(f"\nThis will:")
        print(f"- Use model: {OPENAI_MODEL}")
        print(f"- Generate {N_STORIES_PER_PROMPT} stories for each prompt type")
        print(f"- Make {total_api_calls} total API calls")
        print(f"- Cost approximately ${estimated_cost:.2f} (assuming $0.03 per GPT-4 call)")
        print(f"\nPrompts being tested:")
        for key, prompt_data in PROMPTS.items():
            print(f"  • {prompt_data['display']}: '{prompt_data['text']}'")
        
        proceed = input("\nProceed? (y/n): ")
        if proceed.lower() != 'y':
            print("Operation cancelled.")
            exit(0)

        print(f"\nStarting story generation and analysis using {OPENAI_MODEL}...")
        analyzer = StoryAnalyzer(OPENAI_API_KEY)
        
        for prompt_key in PROMPTS:
            analyzer.process_multiple_stories(
                n_stories=N_STORIES_PER_PROMPT, 
                prompt_key=prompt_key
            )
        
        analyzer.save_results()
        print("\nAnalysis complete! Results saved to 'story_analysis.json'")
        
        # Create visualization only at the end
        analyzer.plot_current_results()
        print("Visualization saved to 'gender_distribution.png'")
