import os
from dotenv import load_dotenv
import openai
from vertexai.language_models import TextGenerationModel
from google.cloud import translate
import vertexai
import logging
import time
import json
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from typing import List, Dict

load_dotenv()

# Validate environment variables for fallback services
required_env_vars = [
    "OPENAI_API_KEY",
    "VERTEX_PROJECT_ID",
    "VERTEX_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS"
]

# Initialize clients if environment variables exist
openai_client = None
translate_client = None
vertex_model = None

if os.getenv("OPENAI_API_KEY"):
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    translate_client = translate.TranslationServiceClient()

if all(os.getenv(var) for var in ["VERTEX_PROJECT_ID", "VERTEX_LOCATION"]):
    vertexai.init(
        project=os.getenv("VERTEX_PROJECT_ID"),
        location=os.getenv("VERTEX_LOCATION", "us-central1")
    )
    
    # Vertex AI settings
    generation_config = {
        "candidate_count": 1,
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 1,
    }
    
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        ),
    ]
    vertex_model = GenerativeModel("gemini-1.5-pro-002")

def translate_with_openai(text: str) -> tuple[str, dict]:
    """Translate text to Icelandic using OpenAI's GPT-4"""
    if not openai_client:
        raise EnvironmentError("OpenAI client not initialized")
    
    logger = logging.getLogger("translation_logger")
    start_time = time.time()
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Translate the following English text to Icelandic. Only respond with the translation, no explanations: {text}"
        }],
        temperature=0
    )
    
    latency = int((time.time() - start_time) * 1000)
    metadata = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "model": response.model,
        "latency": latency
    }
    
    logger.info(f"OpenAI translation metadata: {json.dumps(metadata, indent=2)}")
    return response.choices[0].message.content.strip(), metadata

def translate_with_vertex(text: str) -> tuple[str, dict]:
    """Translate text to Icelandic using Google's Vertex AI"""
    if not vertex_model:
        raise EnvironmentError("Vertex AI model not initialized")
    
    logger = logging.getLogger("translation_logger")
    start_time = time.time()
    
    try:
        response = vertex_model.generate_content(
            f"Translate the following English text to Icelandic. Only respond with the translation, no explanations: {text}",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        latency = int((time.time() - start_time) * 1000)
        metadata = {
            "model": "gemini-1.5-pro-002",
            "latency": latency
        }
        
        logger.info(f"Vertex translation metadata: {json.dumps(metadata, indent=2)}")
        return response.text, metadata
        
    except Exception as e:
        logger.error(f"Vertex translation error details: {str(e)}")
        raise

def translate_with_google(text: str) -> tuple[str, dict]:
    """Translate text to Icelandic using Google Translate API"""
    if not translate_client:
        raise EnvironmentError("Google Translate client not initialized")
    
    logger = logging.getLogger("translation_logger")
    start_time = time.time()
    
    try:
        parent = f"projects/{os.getenv('VERTEX_PROJECT_ID')}/locations/global"
        response = translate_client.translate_text(
            request={
                "parent": parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": "en-US",
                "target_language_code": "is"
            }
        )
        
        latency = int((time.time() - start_time) * 1000)
        metadata = {
            "model": "google-translate-v3",
            "detected_language": response.translations[0].detected_language_code if hasattr(response.translations[0], 'detected_language_code') else "",
            "latency": latency
        }
        
        logger.info(f"Google translation metadata: {json.dumps(metadata, indent=2)}")
        return response.translations[0].translated_text, metadata
        
    except Exception as e:
        logger.error(f"Google translation error details: {str(e)}")
        raise

def get_fallback_translations(text: str) -> dict:
    """Get translations from all fallback services"""
    logger = logging.getLogger("translation_logger")
    translations = {}
    metadata = {}
    
    for service, translate_func in {
        'openai': translate_with_openai,
        'vertex': translate_with_vertex,
        'google': translate_with_google
    }.items():
        try:
            logger.info(f"Starting fallback translation with {service}")
            translation, service_metadata = translate_func(text)
            translations[service] = translation
            metadata[service] = service_metadata
            
        except Exception as e:
            logger.error(f"Error with {service} translation: {str(e)}")
            translations[service] = f"Error: {str(e)}"
            metadata[service] = {"error": str(e)}
    
    return {
        "translations": translations,
        "metadata": metadata
    }

def get_simple_fallback_translations(text: str) -> Dict[str, str]:
    """Get only translations from all fallback services without metadata"""
    try:
        result = get_fallback_translations(text)
        return result["translations"]
    except Exception as e:
        logging.error(f"Fallback translations failed: {str(e)}")
        return {"error": str(e)}

def get_simple_fallback_translations_list(texts: List[str]) -> List[Dict[str, str]]:
    """Get only translations for a list of texts"""
    return [get_simple_fallback_translations(text) for text in texts]

# Example usage:
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example texts
    texts = [
        "Hello world",
        "This is a test",
        "Machine translation is fascinating"
    ]
    
    # Full metadata version
    results = [get_fallback_translations(text) for text in texts]
    print("\nFull results with metadata:")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Simple version
    translations = get_simple_fallback_translations_list(texts)
    print("\nSimple translations:")
    for original, translated in zip(texts, translations):
        print(f"{original} -> {translated}")
