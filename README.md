# fairytales
Creates the first paragraph of fairytales in multiple languages via OpenAI, and then analyses the gender of the main character 

# Fairytale Gender Analysis

A Python-based analysis tool that generates and analyzes fairytale openings across different cultural contexts (English, French, and Chinese) to examine gender representation patterns using GPT-4.

## Overview

This project uses OpenAI's GPT-4 to:
1. Generate fairytale openings from different cultural prompts
2. Analyze the gender of main characters in these stories
3. Visualize the gender distribution patterns across different cultural contexts

## Key Findings

Based on the analysis visualized in `gender_distribution.png`:

- **Female Representation**:
  - Chinese stories: 81% female protagonists
  - English stories: 70% female protagonists
  - French stories: 62% female protagonists

- **Gender-Neutral Stories**:
  - English and French stories: 28% no specific gender
  - Chinese stories: 14% no specific gender

- **Male Representation**:
  - French stories: 10% male protagonists
  - Chinese stories: 5% male protagonists
  - English stories: 2% male protagonists

## Setup

1. Clone this repository
2. Create a `config.py` file with your OpenAI API key:
OPENAI_API_KEY = "your-api-key-here"

3. Install required dependencies: 
pip install openai matplotlib seaborn numpy


## Usage

Run the script:
python generateStories.py


The script will:
- Generate new stories (API key needed) if no existing data is found
- Load and visualize existing data if `story_analysis.json` exists
- Create a visualization saved as `gender_distribution.png`

## Files

- `generateStories.py`: Main script for story generation and analysis
- `story_analysis.json`: JSON file containing generated stories and their analysis
- `gender_distribution.png`: Visualization of gender distribution analysis
- `config.py`: Configuration file for API key (not included in repository)

## Technical Details

- Uses OpenAI's GPT-4 model for story generation and analysis
- Implements error handling and retry mechanisms for API calls
- Saves results in JSON format for persistence
- Creates visualizations using matplotlib and seaborn
- Supports multiple languages/cultural contexts

## Data Storage

Results are stored in `story_analysis.json` with the following structure:

json
{
"timestamp": "ISO timestamp",
"prompt_key": "language key",
"prompt": "story prompt",
"display": "display name",
"story": "generated story",
"analysis": {
"main_character_gender": "gender classification",
"confidence_score": "confidence value",
"reasoning": "analysis explanation"
},
"model": "model name"
}

## License

MIT License