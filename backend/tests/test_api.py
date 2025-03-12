import pytest
from fastapi.testclient import TestClient
import os

def test_analyze_endpoint(client: TestClient):
    """Test the /analyze endpoint."""
    response = client.post(
        "/analyze",
        json={
            "policy": "Universal Basic Income",
            "languages": ["English", "Spanish"],
            "model_id": "gpt-4",
            "samples_per_language": 1
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["policy"] == "Universal Basic Income"
    assert "timestamp" in data
    assert "results" in data
    assert "visualization_url" in data
    assert "summary_url" in data
    
    # Check results structure
    results = data["results"]
    assert "English" in results
    assert "Spanish" in results
    
    # Check English results (should be support=1 based on our mock)
    assert results["English"]["support"] == 1
    assert results["English"]["oppose"] == 0
    
    # Check Spanish results (should be support=0 based on our mock)
    assert results["Spanish"]["support"] == 0
    assert results["Spanish"]["oppose"] == 1

def test_list_analyses(client: TestClient):
    """Test the /analyses endpoint."""
    # First create an analysis using the analyze endpoint
    response = client.post(
        "/analyze",
        json={
            "policy": "Test Policy",
            "languages": ["English"],
            "model_id": "gpt-4",
            "samples_per_language": 1
        }
    )
    assert response.status_code == 200
    
    # Now test the list endpoint
    response = client.get("/analyses")
    assert response.status_code == 200
    analyses = response.json()
    
    # Should be a list with at least one item
    assert isinstance(analyses, list)
    assert len(analyses) >= 1

def test_invalid_request(client: TestClient):
    """Test error handling for invalid requests."""
    response = client.post(
        "/analyze",
        json={
            "policy": "Universal Basic Income",
            # Missing required fields
            "model_id": "gpt-4"
        }
    )
    assert response.status_code == 422  # Validation error 