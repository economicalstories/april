# APRIL: All Policy Really Is Local

## A Cross-Linguistic Investigation of AI Policy Reasoning

<p align="center">
  <img src="https://github.com/economicalstories/april/blob/main/sample_visualization.png" alt="APRIL visualization example" width="700"/>
</p>

APRIL tests whether large language models express consistent policy preferences across different languages, inspired by Anthea Roberts' foundational work "*Is International Law International?*" (2017).

**Key Question**: Do LLMs say the same thing in different languages when asked about policy topics?

## Motivation

Anthea Roberts demonstrated that supposedly "universal" legal principles are interpreted differently depending on where and in what language they're taught. APRIL extends this critical insight to artificial intelligence:

> *If we rely on LLMs in our policy processes, the advice they provide will be conditioned by the language in which we ask our questions.*

Even when a model like GPT-4o is trained on multilingual data, the representations of concepts in different languages don't map identically. This has profound implications for global AI governance and policy deployment.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- OpenAI API key (can be obtained at [OpenAI's platform](https://platform.openai.com/))
- Internet connection for API calls

### Installation

```bash
# Clone the repository
git clone https://github.com/economicalstories/april.git
cd april

# Install required packages
pip install openai matplotlib

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# Or on Windows:
# set OPENAI_API_KEY=your-api-key-here
```

### Running APRIL

Simply execute the main script:

```bash
python april.py
```

The tool will guide you through the following steps:

1. **Select a policy topic** - Enter any policy (e.g., "Universal Basic Income", "Carbon Tax")
2. **Choose languages** - Select from 12 available languages or test all of them
3. **Set sample size** - The tool will show recommended sample sizes:
   
   - 1 sample per language (quick test) - 1 API call per language
   - 10 samples per language (moderate accuracy) - 10 API calls per language  
   - 100 samples per language (good accuracy) - 100 API calls per language
   - 1000 samples per language (high accuracy) - 1000 API calls per language
   
   You'll then be prompted to enter a numerical value (default: 10)
   
4. **Review total API usage** - Before proceeding, you'll be shown the total number of API calls the analysis will require. We recommend checking OpenAI's current pricing at https://openai.com/pricing before proceeding with large analyses.

5. **View results** - Examine the data in three formats:
   - CSV file with all raw responses
   - Text summary with statistics
   - Visualization showing support rates across languages

## Methodology

For each language and policy topic:

1. The policy question is translated into the target language
2. GPT-4o is prompted to:
   - Explain the policy in one sentence
   - Provide one reason supporting the policy
   - Provide one reason opposing the policy
   - State whether it ultimately supports (1) or opposes (0) the policy
3. Results are collected, analyzed, and visualized

This structured approach ensures consistency across languages while allowing us to detect meaningful variation in policy positions.

## Interpreting Results

It's critical to understand what APRIL results do and do not represent:

- **Not a public opinion poll**: Results don't measure what people in different countries think
- **Not measuring cultural bias**: The variation is in the model itself, not society
- **Testing representation alignment**: Do concepts map consistently across language embeddings?

When you see variance across languages, you're observing inconsistencies in how the model processes and reasons about the same policy in different linguistic contexts.

## Key Findings

Initial experiments reveal that policy support rates can vary dramatically by language:

- Some policies show near-perfect consistency across languages
- Others show dramatic variation (e.g., 80% support in one language, 20% in another)
- Languages with similar linguistic roots often (but not always) show similar patterns

These findings raise important questions about the deployment of LLMs in multilingual policy contexts.

## Implications

The inconsistencies revealed by APRIL have significant implications:

1. **Policy AI Safety**: Models deployed globally may reach different conclusions based on language
2. **Fairness and Equity**: Language-based variations could lead to disparate treatment
3. **AI Governance**: Regulatory frameworks must account for cross-linguistic variations
4. **Research Transparency**: AI research findings may not generalize across languages

## Development

APRIL was developed in a single day using Claude 3.7 Sonnet and Cursor as an experiment in "Vibe coding" - a collaborative human-AI approach to rapid application development. This approach demonstrates how complex, theoretically-grounded tools can be created quickly through effective human-AI collaboration.

## Contributing

Contributions are welcome! Areas for improvement include:

- Adding more language options
- Implementing additional policy metrics
- Creating more sophisticated visualizations
- Expanding the theoretical framework
- Testing in the same language across different LLMs 

## License

MIT License - See LICENSE file for details.

## Citation

If you use APRIL in your research, please cite:

```
@software{april2023,
  author = {PC Hubbard},
  title = {APRIL: All Policy Really Is Local},
  year = {2025},
  url = {https://github.com/economicalstories/april}
}
```

## Acknowledgments

Special thanks to Anthea Roberts for the theoretical inspiration behind this project.