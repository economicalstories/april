import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.openai_service import OpenAIService
from src.services.visualization_service import VisualizationService

def estimate_cost(num_languages: int, samples_per_language: int, model: str) -> float:
    """Estimate the OpenAI API cost based on number of calls."""
    # Approximate costs per call (as of March 2024)
    costs = {
        "gpt-4": 0.03,     # $0.03 per 1K tokens
        "gpt-4o": 0.005,   # $0.005 per 1K tokens
        "gpt-4-turbo": 0.01,  # $0.01 per 1K tokens
        "gpt-3.5-turbo": 0.001  # $0.001 per 1K tokens
    }
    
    total_calls = num_languages * samples_per_language
    # Assuming average of 1K tokens per call
    estimated_cost = total_calls * costs.get(model, 0.03)
    return estimated_cost

def run_analysis(
    policy: str,
    languages: List[str] = ["English", "Spanish", "French", "German"],
    samples_per_language: int = 2,
    model: str = "gpt-4",
    temperature: float = 1.0,
    dry_run: bool = False
) -> None:
    """Run a custom analysis with the specified parameters."""
    
    # Calculate estimated API calls and cost
    total_calls = len(languages) * samples_per_language
    estimated_cost = estimate_cost(len(languages), samples_per_language, model)
    
    # Print analysis parameters
    print("\nAnalysis Parameters:")
    print(f"Policy: {policy}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Samples per language: {samples_per_language}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"\nEstimated API calls: {total_calls}")
    print(f"Estimated cost: ${estimated_cost:.3f}")
    
    if dry_run:
        print("\nDry run completed. Use --no-dry-run to execute the analysis.")
        return
    
    # Initialize services
    openai_service = OpenAIService()  # OpenAIService doesn't take model or temperature params
    
    # Use the correct path structure for output directories
    # Ensure path is relative to the project root, not the backend directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    output_dir = os.path.join(project_root, 'data', 'outputs')
    
    viz_service = VisualizationService(output_dir=output_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Initialize results dictionary
    results = {lang: {"support": 0, "oppose": 0, "error": 0, "pros": [], "cons": []} for lang in languages}
    
    print("\nStarting analysis...")
    
    # Run analysis for each language and sample
    for lang in languages:
        print(f"\nAnalyzing policy in {lang}...")
        for _ in range(samples_per_language):
            try:
                response = openai_service.analyze_policy(policy, lang, model)  # Pass model_id as third parameter
                
                # Update results
                if response["support"] == 1:  # Changed from stance to support
                    results[lang]["support"] += 1
                else:
                    results[lang]["oppose"] += 1
                
                # Store unique pros and cons
                if response.get("pro") and response["pro"] not in results[lang]["pros"]:
                    results[lang]["pros"].append(response["pro"])
                if response.get("con") and response["con"] not in results[lang]["cons"]:
                    results[lang]["cons"].append(response["con"])
                        
            except Exception as e:
                print(f"Error in analysis for {lang}: {str(e)}")
                results[lang]["error"] += 1
    
    # Create visualization and summary
    viz_file = viz_service.create_visualization(results, policy, timestamp, model)
    summary_file = viz_service.create_summary(results, policy, timestamp, model, samples_per_language)
    
    print("\nAnalysis completed!")
    print(f"Visualization saved to: {viz_file}")
    print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Run a custom policy analysis")
    parser.add_argument("policy", type=str, help="The policy to analyze")
    parser.add_argument("--languages", type=str, nargs="+", 
                      default=["English", "Spanish", "French", "German"],
                      help="List of languages to analyze (default: English Spanish French German)")
    parser.add_argument("--samples", type=int, default=2,
                      help="Number of samples per language (default: 2)")
    parser.add_argument("--model", type=str, default="gpt-4",
                      choices=["gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                      help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for model responses (default: 1.0)")
    parser.add_argument("--no-dry-run", action="store_true",
                      help="Execute the analysis (default: dry run only)")
    
    args = parser.parse_args()
    
    run_analysis(
        policy=args.policy,
        languages=args.languages,
        samples_per_language=args.samples,
        model=args.model,
        temperature=args.temperature,
        dry_run=not args.no_dry_run
    )

if __name__ == "__main__":
    main() 