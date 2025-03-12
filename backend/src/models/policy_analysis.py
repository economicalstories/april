from typing import Dict, List, Optional, Any
from datetime import datetime

class PolicyAnalysis:
    def __init__(self, policy: str, model_name: str, timestamp: str = None):
        self.policy = policy
        self.model_name = model_name
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d%H%M%S")
        self.results: Dict[str, Dict[str, Any]] = {}
        self.samples_per_language: int = 0

    def add_result(self, language: str, support: bool, explanation: str, pro: str, con: str):
        if language not in self.results:
            self.results[language] = {
                'support': 0,
                'oppose': 0,
                'error': 0,
                'pros': [],
                'cons': []
            }
        
        if support:
            self.results[language]['support'] += 1
            if pro:
                self.results[language]['pros'].append(pro)
        else:
            self.results[language]['oppose'] += 1
            if con:
                self.results[language]['cons'].append(con)

    @property
    def safe_policy_name(self) -> str:
        """Return a filesystem-safe version of the policy name."""
        return self.policy.lower().replace(' ', '_').replace('-', '_') 