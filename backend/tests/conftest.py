import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import shutil
from typing import Generator
from dotenv import load_dotenv

# Load test environment variables
test_env_path = os.path.join(os.path.dirname(__file__), '.env.test')
load_dotenv(test_env_path)

from src.api.main import app
from src.services.visualization_service import VisualizationService
from src.services.openai_service import OpenAIService

@pytest.fixture
def test_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    # Set the environment variable for the visualization service
    os.environ["TEST_OUTPUT_DIR"] = temp_dir
    yield temp_dir
    shutil.rmtree(temp_dir)
    # Clean up the environment variable
    if "TEST_OUTPUT_DIR" in os.environ:
        del os.environ["TEST_OUTPUT_DIR"]

@pytest.fixture
def viz_service(test_output_dir):
    """Create a VisualizationService instance with test directory."""
    return VisualizationService(output_dir=test_output_dir)

@pytest.fixture
def mock_openai_service(monkeypatch):
    """Create a mock OpenAI service that returns predictable results."""
    class MockOpenAI:
        def translate_prompt(self, policy: str, language: str, model_id: str) -> str:
            return f"Test prompt for {policy} in {language}"

        def analyze_policy(self, prompt: str, language_code: str, model_id: str) -> dict:
            return {
                "explanation": f"Test explanation in {language_code}",
                "pro": "Test pro argument",
                "con": "Test con argument",
                "support": 1 if language_code.lower() == "english" else 0
            }

    monkeypatch.setattr("src.api.main.openai_service", MockOpenAI())
    return MockOpenAI()

@pytest.fixture
def client(mock_openai_service, viz_service) -> Generator:
    """Create a TestClient instance with mocked services."""
    with TestClient(app) as test_client:
        yield test_client 