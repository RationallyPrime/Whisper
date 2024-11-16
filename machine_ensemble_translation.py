import os
from dotenv import load_dotenv
import anthropic
from typing import List, Dict, Optional
import json
import time
from datetime import datetime
from pydantic import BaseModel
import logging
from pathlib import Path
import csv
import yaml
import pandas as pd
from markdown_it import MarkdownIt
from functools import lru_cache
import hashlib
import re

load_dotenv()

# Validate required environment variables
if not os.getenv("ANTHROPIC_API_KEY"):
    raise EnvironmentError("Missing required environment variable: ANTHROPIC_API_KEY")

# Initialize Claude client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

from pydantic import BaseModel, Field
from typing import Optional

class PhaseSpecification(BaseModel):
    number: int = Field(..., description="Númer")
    name: str = Field(..., description="Heiti verkþáttar")
    responsible_party: str = Field(..., description="Ábyrgur aðili")
    start_month: Optional[str] = Field(None, description="Upphafsmánuður")
    end_month: Optional[str] = Field(None, description="Lokamánuður")
    cost_percentage: float = Field(default=0, description="Hlutfall af heildarkostnaði verkefnis (%)")
    description: str = Field(..., description="Lýsið verkþætti", max_length=1000)
    subtasks: str = Field(..., description="Lýsið undirverkþáttum og hverjir koma að þeim", max_length=1000)
    deliverables: str = Field(..., description="Lýsið vörum og niðurstöðum", max_length=1000)
    
# Core translation functionality
class TranslationCache:
    def __init__(self, glossary_path: str | Path):
        self.glossary = self._load_glossary(glossary_path)
        self.glossary_context = self._prepare_glossary_context()
        self._cache = {}  # In-memory cache for translations
    
    def _load_glossary(self, path: str | Path) -> dict:
        """Load technical glossary from markdown file with table format"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            glossary = {}
            
            # Split content into sections
            sections = content.split('##')
            
            for section in sections:
                if '|' not in section:  # Skip non-table sections
                    continue
                    
                # Process each table
                lines = section.strip().split('\n')
                header = lines[0].strip()
                
                # Skip header row and separator row
                table_rows = [row for row in lines if '|' in row]
                table_rows = table_rows[2:]  # Skip header and separator
                
                for row in table_rows:
                    cells = [cell.strip() for cell in row.split('|')]
                    if len(cells) >= 3:  # Ensure we have at least English and Icelandic
                        english = cells[1].strip()
                        icelandic = cells[2].strip()
                        if english and icelandic:
                            glossary[english] = icelandic
            
            return glossary

    def _prepare_glossary_context(self) -> str:
        """Prepare glossary context string with sections"""
        context = []
        context.append("# Technical Translation Guidelines")
        
        # Core Technical Terms
        context.append("\n## Core Technical Terms")
        terms = [f"- {en}: {is_}" for en, is_ in self.glossary.items()]
        context.extend(terms)
        
        # Add restructuring patterns
        context.append("\n## Common Phrase Restructuring")
        context.append("When translating, apply these patterns:")
        context.append("- Use active voice in technical descriptions")
        context.append("- Use passive voice in formal documentation")
        context.append("- Prefer natural Icelandic expressions over direct translations")
        
        return "\n".join(context)

    def get_relevant_terms(self, text: str) -> str:
        """Get glossary terms relevant to the given text"""
        relevant_terms = []
        for term, translation in self.glossary.items():
            if term.lower() in text.lower():
                relevant_terms.append(f"- {term}: {translation}")
        return "\n".join(relevant_terms)

def translate_with_claude(text: str, cache: TranslationCache) -> tuple[str, dict]:
    """Enhanced translation function with glossary context and prompt caching"""
    
    # Prepare glossary context
    glossary_terms = cache.get_relevant_terms(text)
    
    # Create the system content array with both prompt and glossary
    system = [
        {
            "type": "text",
            "text": """Translate the following English text to Icelandic, adhering to these guidelines:

1. DOMAIN CONTEXT:
This is a technical grant application for a legal technology project. Maintain formal academic/technical tone appropriate for grant reviewers.

2. STRUCTURAL GUIDELINES:
- Preserve paragraph structure but adapt sentence structure to natural Icelandic flow
- Convert bullet points into flowing text where appropriate
- Maintain any specific formatting (headers, lists where necessary)
- Preserve any numerical references or citations exactly

3. TRANSLATION PRIORITIES:
- Prefer natural Icelandic expressions over direct translations
- Maintain consistent terminology throughout sections
- Use active voice for technical descriptions
- Use passive voice for formal documentation
- Restructure marketing concepts into natural Icelandic patterns

4. GLOSSARY ADHERENCE:
- Strictly use the provided technical term translations
- Apply restructuring patterns for business concepts
- Maintain consistency with previously translated sections

5. OUTPUT FORMAT:
Provide only the translated text, maintaining original formatting and structure where appropriate. Do not include explanations or alternatives.""",
            "cache_control": {"type": "ephemeral"}
        },
        {
            "type": "text",
            "text": f"Technical glossary for reference:\n{cache.glossary_context}",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": text}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
    )
    
    translation = response.content[0].text.strip()
    metadata = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": getattr(response.usage, 'cache_read_input_tokens', 0),
        "cache_write_tokens": getattr(response.usage, 'cache_creation_input_tokens', 0)
    }
    
    return translation, metadata

def translate_text(text: str, cache: TranslationCache) -> dict:
    """Get translation from Claude"""
    logger = logging.getLogger("translation_logger")
    try:
        translation, metadata = translate_with_claude(text, cache)
        return {
            "translations": {"claude": translation},
            "metadata": {"claude": metadata}
        }
    except Exception as e:
        logger.error(f"Error with Claude translation: {str(e)}")
        return {
            "translations": {"claude": f"Error: {str(e)}"},
            "metadata": {"claude": {"error": str(e)}}
        }

def translate_simple_text(text: str, cache: TranslationCache) -> str:
    """Translate a single text and return only the translation"""
    try:
        result = translate_text(text, cache)
        return result["translations"]["claude"]
    except Exception as e:
        logging.error(f"Translation failed: {str(e)}")
        return f"Error: {str(e)}"

def translate_text_list(texts: List[str], cache: TranslationCache) -> List[dict]:
    """Translate a list of texts using Claude with prompt caching"""
    translations = []
    total_cache_hits = 0
    total_tokens = 0
    
    for text in texts:
        result = translate_text(text, cache)
        translations.append(result)
        
        # Track cache performance
        metadata = result["metadata"]["claude"]
        if "cache_read_tokens" in metadata:
            total_cache_hits += metadata["cache_read_tokens"]
        total_tokens += metadata["input_tokens"]
    
    # Log cache performance
    if total_tokens > 0:
        cache_hit_rate = (total_cache_hits / total_tokens) * 100
        logging.info(f"Cache hit rate: {cache_hit_rate:.2f}%")
    
    return translations

def translate_simple_text_list(texts: List[str], cache: TranslationCache) -> List[str]:
    """Translate a list of texts and return only the translations"""
    return [translate_simple_text(text, cache) for text in texts]

# File reading utilities
def read_text_file(file_path: str | Path) -> List[str]:
    """Read lines from a plain text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def read_json_file(file_path: str | Path, text_key: str) -> List[str]:
    """Read texts from a JSON file using specified key"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return [item[text_key] for item in data if text_key in item]
        return [data[text_key]] if text_key in data else []

def read_csv_file(file_path: str | Path, text_column: str) -> List[str]:
    """Read texts from a CSV file using specified column"""
    df = pd.read_csv(file_path)
    return df[text_column].dropna().tolist()

def read_yaml_file(file_path: str | Path, text_key: str) -> List[str]:
    """Read texts from a YAML file using specified key"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        if isinstance(data, list):
            return [item[text_key] for item in data if text_key in item]
        return [data[text_key]] if text_key in data else []

def read_markdown_file(
    file_path: str | Path,
    heading_level: Optional[int] = None,
    include_code_blocks: bool = False,
    start_heading: Optional[str] = None,
    end_heading: Optional[str] = None
) -> List[str]:
    """Read text from a markdown file with various filtering options"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    md = MarkdownIt()
    tokens = md.parse(content)
    texts = []
    current_level = 0
    include_text = start_heading is None
    
    for token in tokens:
        if token.type == 'heading_open':
            current_level = int(token.tag[1])
            next_token = tokens[tokens.index(token) + 1]
            heading_text = next_token.content
            
            if start_heading and heading_text == start_heading:
                include_text = True
            elif end_heading and heading_text == end_heading:
                include_text = False
            
            if heading_level is None or current_level == heading_level:
                texts.append(heading_text)
        
        elif token.type == 'text' and include_text:
            if heading_level is None or current_level == heading_level:
                if token.content.strip():
                    texts.append(token.content.strip())
        
        elif token.type == 'fence' and include_code_blocks and include_text:
            if token.content.strip():
                texts.append(token.content.strip())
    
    return texts

# File translation functionality
def translate_file(
    file_path: str | Path,
    cache: TranslationCache,
    output_path: str | Path = None,
    text_key: str = None,
    include_metadata: bool = False,
    markdown_options: Dict = None
) -> None:
    """Translate texts from a file with prompt caching and save results"""
    file_path = Path(file_path)
    if not output_path:
        output_path = file_path.parent / f"translated_{file_path.name}"
    
    # For markdown files, read and process the entire content
    if file_path.suffix.lower() in ['.md', '.markdown']:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into larger chunks that preserve context (roughly 2000 chars each)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in content.split('\n\n'):
            if current_length + len(paragraph) > 2000:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Translate each chunk while preserving code blocks
        translated_chunks = []
        for chunk in chunks:
            # Temporarily replace code blocks with placeholders
            code_blocks = []
            chunk_without_code = chunk
            
            code_pattern = r'```[^\n]*\n.*?```'
            for match in re.finditer(code_pattern, chunk, re.DOTALL):
                code_block = match.group(0)
                placeholder = f'CODE_BLOCK_{len(code_blocks)}'
                code_blocks.append(code_block)
                chunk_without_code = chunk_without_code.replace(code_block, placeholder)
            
            # Translate the chunk
            translated_chunk = translate_simple_text(chunk_without_code, cache)
            
            # Restore code blocks
            for i, code_block in enumerate(code_blocks):
                translated_chunk = translated_chunk.replace(f'CODE_BLOCK_{i}', code_block)
            
            translated_chunks.append(translated_chunk)
        
        # Write the complete translated content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(translated_chunks))
    else:
        # Original handling for non-markdown files
        # Read texts based on file type
        suffix = file_path.suffix.lower()
        if suffix == '.txt':
            texts = read_text_file(file_path)
        elif suffix == '.md' or suffix == '.markdown':
            markdown_options = markdown_options or {}
            texts = read_markdown_file(file_path, **markdown_options)
        elif suffix == '.json':
            texts = read_json_file(file_path, text_key)
        elif suffix == '.csv':
            texts = read_csv_file(file_path, text_key)
        elif suffix in ['.yml', '.yaml']:
            texts = read_yaml_file(file_path, text_key)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # Track translation start time
        start_time = time.time()
        
        # Translate texts with caching
        if include_metadata:
            translations = translate_text_list(texts, cache)
        else:
            translations = translate_simple_text_list(texts, cache)
        
        # Calculate and log translation time
        elapsed_time = time.time() - start_time
        logging.info(f"Translation completed in {elapsed_time:.2f} seconds")
        
        # Save translations
        with open(output_path, 'w', encoding='utf-8') as f:
            if include_metadata:
                json.dump({
                    'source_file': str(file_path),
                    'translations': translations,
                    'performance': {
                        'translation_time': elapsed_time
                    }
                }, f, ensure_ascii=False, indent=2)
            else:
                if suffix == '.txt':
                    f.write('\n'.join(translations))
                else:
                    json.dump(translations, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Translations saved to: {output_path}")

# Phase translation functionality
def translate_phase(phase: PhaseSpecification, cache: TranslationCache) -> dict:
    """Translate a phase specification"""
    translations = {
        'name': translate_simple_text(phase.name, cache),
        'description': translate_simple_text(phase.description, cache),
        'tasks': []
    }
    
    for task in phase.tasks:
        task_translation = {
            'task_id': task.task_id,
            'description': translate_simple_text(task.description, cache),
            'deliverables': [
                translate_simple_text(deliverable, cache) 
                for deliverable in task.deliverables
            ]
        }
        translations['tasks'].append(task_translation)
    
    return translations

# Main execution
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize cache with glossary
    cache = TranslationCache("technical_glossary.md")
    
    # Track start time
    start_time = time.time()
    
    # Define the directory and files
    input_dir = Path("/home/rationallyprime/Downloads/rannis")
    phase_file = input_dir / "Socrates_phases.json"
    markdown_file = input_dir / "grant_sections.md"
    
    # Process phase specifications
    try:
        with open(phase_file, 'r', encoding='utf-8') as f:
            phases_data = json.load(f)
        
        # Translate each phase specification
        translated_phases = []
        for phase in phases_data:
            phase_spec = PhaseSpecification(**phase)
            translated_phase = {
                "number": phase_spec.number,
                "name": translate_simple_text(phase_spec.name, cache),
                "responsible_party": phase_spec.responsible_party,
                "start_month": phase_spec.start_month,
                "end_month": phase_spec.end_month,
                "cost_percentage": phase_spec.cost_percentage,
                "description": translate_simple_text(phase_spec.description, cache),
                "subtasks": translate_simple_text(phase_spec.subtasks, cache),
                "deliverables": translate_simple_text(phase_spec.deliverables, cache)
            }
            translated_phases.append(translated_phase)
            
        # Save translated phases
        output_phase_file = input_dir / "Socrates_phases-translated.json"
        with open(output_phase_file, 'w', encoding='utf-8') as f:
            json.dump(translated_phases, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully translated phases file: {output_phase_file}")
        
    except Exception as e:
        logging.error(f"Error translating phases file: {str(e)}")
    
    # Process markdown file
    try:
        output_md_file = input_dir / "grant_sections-translated.md"
        translate_file(
            file_path=markdown_file,
            cache=cache,
            output_path=output_md_file,
            markdown_options={
                "include_code_blocks": True
            }
        )
        logging.info(f"Successfully translated markdown file: {output_md_file}")
    except Exception as e:
        logging.error(f"Error translating markdown file: {str(e)}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logging.info(f"All translations completed in {elapsed_time:.2f} seconds")