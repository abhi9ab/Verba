"""
NLP tasks module implementing core functionalities like summarization,
sentiment analysis, NER, and question answering using the LLM.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from llm_abstraction import LLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPProcessor:
    """Class for handling various NLP tasks"""
    
    def __init__(self, llm: LLMInterface):
        """
        Initialize the NLP processor with an LLM.
        
        Args:
            llm: The language model implementation to use
        """
        self.llm = llm
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            Text summary
        """
        logger.info(f"Summarizing text of length {len(text)}")
        
        prompt = f"""Please provide a concise summary of the following text. The summary should capture the main points and key information:

{text}

Summary:"""
        
        try:
            return self.llm.generate(prompt, max_length=max_length)
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return "Failed to generate summary."
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze the sentiment of the input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment class and confidence
        """
        logger.info(f"Analyzing sentiment of text: length {len(text)}")
        
        prompt = f"""Analyze the sentiment of the following text and categorize it as POSITIVE, NEGATIVE, or NEUTRAL. Provide a confidence score between 0 and 1:

{text}

Sentiment analysis:
- Sentiment: 
- Confidence: 
- Explanation:"""
        
        try:
            response = self.llm.generate(prompt, max_length=200)
            
            # Parse the response to extract sentiment and confidence
            sentiment = "NEUTRAL"  # Default value
            confidence = 0.5  # Default value
            explanation = ""
            
            for line in response.split('\n'):
                if "Sentiment:" in line:
                    sentiment_part = line.split("Sentiment:")[1].strip().upper()
                    if "POSITIVE" in sentiment_part:
                        sentiment = "POSITIVE"
                    elif "NEGATIVE" in sentiment_part:
                        sentiment = "NEGATIVE"
                    elif "NEUTRAL" in sentiment_part:
                        sentiment = "NEUTRAL"
                
                if "Confidence:" in line:
                    confidence_part = line.split("Confidence:")[1].strip()
                    # Extract the first float-like string
                    import re
                    confidence_matches = re.findall(r"0\.\d+|\d+\.\d+|\d+", confidence_part)
                    if confidence_matches:
                        confidence = float(confidence_matches[0])
                        # Ensure it's between 0 and 1
                        confidence = max(0.0, min(1.0, confidence))
                
                if "Explanation:" in line:
                    explanation = line.split("Explanation:")[1].strip()
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "sentiment": "NEUTRAL",
                "confidence": 0.0,
                "explanation": f"Error analyzing sentiment: {str(e)}"
            }
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from the input text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of dictionaries with entity text and type
        """
        logger.info(f"Extracting entities from text: length {len(text)}")
        
        prompt = f"""Extract named entities from the following text. For each entity, identify its type (PERSON, LOCATION, ORGANIZATION, DATE, etc.) and the text of the entity:

{text}

Named entities (in JSON format):
```
[
  {{"entity": "entity text", "type": "ENTITY_TYPE"}},
  ...
]
```"""
        
        try:
            response = self.llm.generate(prompt, max_length=1000)
            
            # Extract JSON part from the response
            import re
            import json
            
            json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
            if json_match:
                json_str = json_match.group(1)
                entities = json.loads(json_str)
                return entities
            else:
                # Fallback: try to find any JSON array in the response
                json_pattern = r'\[\s*\{.*?\}\s*\]'
                json_match = re.search(json_pattern, response, re.DOTALL)
                if json_match:
                    try:
                        entities = json.loads(json_match.group(0))
                        return entities
                    except json.JSONDecodeError:
                        pass
                
                logger.warning("Could not extract properly formatted entities JSON from response")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def answer_question(self, context: str, question: str) -> str:
        """
        Answer a question based on the provided context.
        
        Args:
            context: Context information to base the answer on
            question: Question to answer
            
        Returns:
            Answer to the question
        """
        logger.info(f"Answering question based on context: length {len(context)}")
        
        prompt = f"""Use the following context to answer the question. If the answer cannot be found in the context, state that explicitly.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            return self.llm.generate(prompt, max_length=300)
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "Failed to generate an answer."
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """
        Generate code based on a description.
        
        Args:
            description: Description of the code to generate
            language: Programming language to generate code in
            
        Returns:
            Generated code
        """
        logger.info(f"Generating {language} code for description: {description[:50]}...")
        
        prompt = f"""Generate {language} code for the following description. Include comments explaining the key parts of the code:

Description: {description}

```{language}
"""
        
        try:
            response = self.llm.generate(prompt, max_length=800)
            
            # Ensure the response contains properly formatted code
            if "```" in response:
                # Extract content from code block
                import re
                code_match = re.search(r'```(?:' + re.escape(language) + r')?\s*([\s\S]*?)```', response)
                if code_match:
                    return code_match.group(1).strip()
            
            # If no code block detected, return as is (removing only the final backticks if present)
            return response.rstrip('`')
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return f"# Failed to generate {language} code.\n# Error: {str(e)}"