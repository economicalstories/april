import pytest
import os
from src.services.visualization_service import VisualizationService
from src.models.policy_analysis import PolicyAnalysis

def test_visualization_service(test_output_dir):
    """Test the visualization service."""
    service = VisualizationService(output_dir=test_output_dir)
    
    # Test directory creation
    assert os.path.exists(service.viz_dir)
    assert os.path.exists(service.analyses_dir)
    assert os.path.exists(service.summaries_dir)
    
    # Create test data
    results = {
        "English": {
            "support": 7,
            "oppose": 3,
            "error": 0,
            "pros": ["Good idea"],
            "cons": ["Too expensive"]
        },
        "Spanish": {
            "support": 3,
            "oppose": 7,
            "error": 0,
            "pros": ["Buena idea"],
            "cons": ["Muy caro"]
        }
    }
    
    # Test visualization creation
    viz_file = service.create_visualization(
        results=results,
        policy="Test Policy",
        timestamp="20250101000000",
        model_name="test-model"
    )
    assert os.path.exists(viz_file)
    assert viz_file.endswith(".png")
    
    # Test summary creation
    summary_file = service.create_summary(
        results=results,
        policy="Test Policy",
        timestamp="20250101000000",
        model_name="test-model",
        samples_per_language=10
    )
    assert os.path.exists(summary_file)
    assert summary_file.endswith(".txt")
    
    # Check summary content
    with open(summary_file, 'r') as f:
        content = f.read()
        assert "Test Policy" in content
        assert "test-model" in content
        assert "English" in content
        assert "Spanish" in content

def test_policy_analysis():
    """Test the PolicyAnalysis model."""
    analysis = PolicyAnalysis(
        policy="Test Policy",
        model_name="test-model",
        timestamp="20250101000000"
    )
    
    # Test initialization
    assert analysis.policy == "Test Policy"
    assert analysis.model_name == "test-model"
    assert analysis.timestamp == "20250101000000"
    assert isinstance(analysis.results, dict)
    
    # Test adding results
    analysis.add_result(
        language="English",
        support=True,
        explanation="Test explanation",
        pro="Test pro",
        con="Test con"
    )
    
    assert "English" in analysis.results
    assert analysis.results["English"]["support"] == 1
    assert analysis.results["English"]["oppose"] == 0
    assert len(analysis.results["English"]["pros"]) == 1
    assert analysis.results["English"]["pros"][0] == "Test pro"
    
    # Test safe policy name
    assert analysis.safe_policy_name == "test_policy"
    
    # Test with spaces and hyphens
    analysis = PolicyAnalysis(
        policy="Test-Policy With Spaces",
        model_name="test-model"
    )
    assert analysis.safe_policy_name == "test_policy_with_spaces" 