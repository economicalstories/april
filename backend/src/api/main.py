from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import csv

from ..services.openai_service import OpenAIService
from ..services.visualization_service import VisualizationService
from ..models.policy_analysis import PolicyAnalysis

app = FastAPI(title="APRIL API", description="API for All Policy Really Is Local")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
openai_service = OpenAIService()

# Create a function to get the visualization service
def get_viz_service():
    """Get the visualization service with the appropriate output directory."""
    output_dir = os.getenv("TEST_OUTPUT_DIR", "data/outputs")
    return VisualizationService(output_dir=output_dir)

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    policy: str
    languages: List[str]
    model_id: str
    samples_per_language: int

class AnalysisResponse(BaseModel):
    policy: str
    timestamp: str
    results: Dict
    visualization_url: str
    summary_url: str

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_policy(request: AnalysisRequest, viz_service: VisualizationService = Depends(get_viz_service)):
    """Analyze a policy across multiple languages."""
    try:
        analysis = PolicyAnalysis(
            policy=request.policy,
            model_name=request.model_id
        )
        analysis.samples_per_language = request.samples_per_language

        # Create CSV file for analysis results
        csv_file = os.path.join(
            viz_service.analyses_dir,
            f"{analysis.safe_policy_name}_analysis_{analysis.timestamp}.csv"
        )
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['language', 'sentiment', 'support', 'explanation', 'pro', 'con'])

            for language in request.languages:
                for _ in range(request.samples_per_language):
                    # Get the analysis directly in the target language
                    result = openai_service.analyze_policy(
                        request.policy,
                        language,
                        request.model_id
                    )

                    # Add to analysis
                    analysis.add_result(
                        language=language,
                        support=bool(result['support']),
                        explanation=result['explanation'],
                        pro=result['pro'],
                        con=result['con']
                    )

                    # Write to CSV
                    writer.writerow([
                        language,
                        'support' if result['support'] else 'oppose',
                        result['support'],  # Numeric support value (0 or 1)
                        result['explanation'],
                        result['pro'],
                        result['con']
                    ])

        # Generate visualization and summary
        viz_file = viz_service.create_visualization(
            analysis.results,
            analysis.policy,
            analysis.timestamp,
            request.model_id
        )

        summary_file = viz_service.create_summary(
            analysis.results,
            analysis.policy,
            analysis.timestamp,
            request.model_id,
            request.samples_per_language
        )

        return AnalysisResponse(
            policy=analysis.policy,
            timestamp=analysis.timestamp,
            results=analysis.results,
            visualization_url=f"/outputs/visualizations/{os.path.basename(viz_file)}",
            summary_url=f"/outputs/summaries/{os.path.basename(summary_file)}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyses")
async def list_analyses(viz_service: VisualizationService = Depends(get_viz_service)):
    """List all available analyses."""
    try:
        analyses = []
        if os.path.exists(viz_service.analyses_dir):
            for filename in os.listdir(viz_service.analyses_dir):
                if filename.endswith('.csv'):
                    policy_name = filename.split('_analysis_')[0].replace('_', ' ').title()
                    timestamp = filename.split('_analysis_')[1].split('.')[0]
                    analyses.append({
                        "policy": policy_name,
                        "timestamp": timestamp,
                        "file": filename
                    })
        return analyses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 