import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import time
from typing import Dict, Any, Tuple
import csv
import os.path

class OpenAIService:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # Minimum time between requests in seconds
        
        # Create output directories
        os.makedirs("data/outputs/translations", exist_ok=True)
        
        # Cache for translated user prompts
        self.translation_cache = {}
        
        # Standard system prompt (always in English, used for all languages)
        self.system_prompt = """You are a policy analyst. First understand and explain the policy, then analyze its implications, and finally make a decision on your stance.
Respond in exactly this format:

EXPLANATION: [One clear sentence explaining what the policy means]
PRO: [One major argument in favor]
CON: [One major argument against]
STANCE: [1 or 0]

Your response must follow these rules:
1. Use exactly this format with these field names
2. First explain the policy clearly, then give balanced pro/con analysis
3. Only after considering the explanation and arguments, decide on your stance
4. For STANCE, use exactly 1 for support or 0 for oppose
5. Keep each response to one clear sentence
6. Do not include any other text"""

        # Additional instructions for non-English languages (appended to system prompt)
        self.non_english_instructions = """

IMPORTANT: Keep field names in English (EXPLANATION, PRO, CON, STANCE) but write your analysis in the language requested. The policy statement will be provided in that language."""

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()
    
    def translate_text(self, text: str, language: str, model_id: str) -> str:
        """Translate text into the target language."""
        # Skip translation for English
        if language.lower() == 'english':
            return text
            
        print(f"Translating to {language}...")
        self._wait_for_rate_limit()
        
        try:
            # Translation prompt
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional translator. Translate the following text into {language} accurately and naturally. Provide ONLY the translation with no explanations, notes, or other text."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            
            translated_text = response.choices[0].message.content.strip()
            return translated_text
            
        except Exception as e:
            print(f"Error in translation to {language}: {str(e)}")
            # If translation fails, return original text
            return text
    
    def get_analysis_prompts(self, policy: str, language: str, model_id: str) -> Tuple[str, str]:
        """Generate system and user prompts for policy analysis."""
        # Check if we already have a cached translation for this policy and language
        cache_key = f"{policy}_{language}_{model_id}"
        if cache_key in self.translation_cache:
            print(f"Using cached prompt for {language}")
            return self.translation_cache[cache_key]
        
        # Create the appropriate system prompt (with non-English instructions if needed)
        if language.lower() == 'english':
            system_prompt = self.system_prompt
        else:
            system_prompt = self.system_prompt + self.non_english_instructions
        
        # English user prompt - simple and direct
        english_user_prompt = f"Please indicate your support or opposition to this policy: {policy}"
        
        # For English, we don't translate the user prompt
        if language.lower() == 'english':
            print(f"Using English prompts for: {policy}")
            result = (system_prompt, english_user_prompt)
            self.translation_cache[cache_key] = result
            return result
        
        # For non-English, translate only the user prompt
        print(f"Preparing prompt in {language} (will only happen once)...")
        translated_user_prompt = self.translate_text(english_user_prompt, language, model_id)
        
        # Display the translated prompt for verification
        print(f"English prompt: {english_user_prompt}")
        print(f"Translated prompt ({language}): {translated_user_prompt}")
        
        # Cache the result
        result = (system_prompt, translated_user_prompt)
        self.translation_cache[cache_key] = result
        return result

    def analyze_policy(self, policy: str, language: str, model_id: str) -> Dict[str, Any]:
        """Analyze a policy using OpenAI's API."""
        print(f"Analyzing policy in {language}...")
        
        try:
            # Get the system and user prompts (will use cache if available)
            system_prompt, user_prompt = self.get_analysis_prompts(policy, language, model_id)
            
            self._wait_for_rate_limit()
            
            # Make the API call with both system and user prompts
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            result = response.choices[0].message.content.strip()
            print(f"Received response for {language}: {result[:50]}...")
            
            # Split the response into lines and create a dictionary
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            response_dict = {}
            
            # More robust field extraction - try different approaches
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().upper()  # Normalize keys to uppercase
                    if key in ['EXPLANATION', 'PRO', 'CON', 'STANCE']:
                        response_dict[key] = parts[1].strip()
            
            # If standard parsing failed, try more aggressive approach for STANCE at least
            if 'STANCE' not in response_dict:
                # Look for standalone digits or words like "1", "0", "STANCE: 1", etc.
                for line in lines:
                    if '1' in line or '0' in line:
                        for char in line:
                            if char in ['0', '1']:
                                response_dict['STANCE'] = char
                                break
                        if 'STANCE' in response_dict:
                            break
                
                # If still no stance, check the whole response for the digit
                if 'STANCE' not in response_dict:
                    if '1' in result:
                        response_dict['STANCE'] = '1'
                    elif '0' in result:
                        response_dict['STANCE'] = '0'
            
            # Ensure all required fields are present
            required_fields = {'EXPLANATION', 'PRO', 'CON', 'STANCE'}
            missing_fields = required_fields - response_dict.keys()
            
            if missing_fields:
                print(f"Missing fields in {language} response: {missing_fields}")
                # Try to extract at least some content for missing fields
                if 'EXPLANATION' not in response_dict and len(lines) > 0:
                    response_dict['EXPLANATION'] = lines[0]
                if 'PRO' not in response_dict and len(lines) > 1:
                    response_dict['PRO'] = lines[1]
                if 'CON' not in response_dict and len(lines) > 2:
                    response_dict['CON'] = lines[2]
                
                # If we still have missing required fields and couldn't recover them
                missing_fields = required_fields - response_dict.keys()
                if missing_fields:
                    print(f"Could not recover fields: {missing_fields}")
                    # Fill in missing fields with error messages
                    for field in missing_fields:
                        response_dict[field] = f"Error: missing {field.lower()} field"
            
            try:
                stance = int(response_dict.get('STANCE', '0'))
                if stance not in [0, 1]:
                    print(f"Invalid stance value in {language} response: {stance}")
                    stance = 0
            except (ValueError, KeyError):
                print(f"Error parsing stance in {language} response")
                stance = 0
            
            return {
                "explanation": response_dict.get('EXPLANATION', "Error: missing explanation"),
                "support": stance,
                "pro": response_dict.get('PRO', "Error: missing pro argument"),
                "con": response_dict.get('CON', "Error: missing con argument"),
                "user_prompt": user_prompt,  # Only include the user prompt, not system prompt
                "raw_response": result  # Include the raw response for debugging
            }
            
        except Exception as e:
            print(f"Error in OpenAI API call for {language}: {str(e)}")
            return {
                "explanation": "Error in analysis",
                "support": 0,
                "pro": "Error occurred",
                "con": "Error occurred",
                "user_prompt": "",
                "raw_response": ""
            } 