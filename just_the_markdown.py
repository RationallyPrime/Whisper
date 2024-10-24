import json
import os
from datetime import datetime
import logging
import asyncio
from anthropic import AsyncAnthropic  # Changed to AsyncAnthropic
from typing import Dict, Any
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
import tiktoken

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('summary_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Update constants for larger files and rate limiting
MAX_CHUNK_SIZE = 25000  # Increased chunk size
RATE_LIMIT = 10  # 100 requests per minute = 1 request per 0.6 seconds
MAX_CONCURRENT_REQUESTS = 1  # Process one at a time due to token limit
BACKOFF_INITIAL = 2  # Initial backoff for rate limit errors
BACKOFF_MAX = 64
BATCH_SIZE = 5  # Smaller batch size for these large files
CLAUDE_MAX_TOKENS = 200000  # Claude 3 Haiku context window
TARGET_CHUNK_TOKENS = 15000  # Target size for each chunk
PROCESSED_FILE = "processed_conversations.json"  # Added this line
TOKENS_PER_MINUTE = 100_000  # API tier limit

class SummaryCollection:
    def __init__(self, output_file: str = "conversation_summaries.json"):
        self.summaries: Dict[str, Dict[str, Any]] = {}
        self.output_file = output_file
        
    def add_summary(self, filename: str, title: str, summary: str, date: str):
        self.summaries[filename] = {
            "title": title,
            "summary": summary,
            "date": date
        }
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(
                dict(sorted(
                    self.summaries.items(),
                    key=lambda x: x[1]['date'],
                    reverse=True
                )),
                f,
                indent=2
            )

class ProcessingTracker:
    def __init__(self, tracking_file: str):
        self.tracking_file = tracking_file
        try:
            with open(self.tracking_file, 'r') as f:
                self.processed = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.processed = {}
    
    def mark_processed(self, filepath: str, metadata: Dict[str, Any]):
        self.processed[filepath] = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        with open(self.tracking_file, 'w') as f:
            json.dump(self.processed, f, indent=2)
    
    def is_processed(self, filepath: str) -> bool:
        return filepath in self.processed

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    encoder = tiktoken.get_encoding("cl100k_base")  # Claude uses similar tokenization to cl100k
    return len(encoder.encode(text))

async def get_summary(conversation_text: str, client: AsyncAnthropic, semaphore: asyncio.Semaphore) -> Dict[str, str]:
    # First count total tokens
    total_tokens = count_tokens(conversation_text)
    logger.info(f"Processing conversation with {total_tokens} tokens")

    if total_tokens > TARGET_CHUNK_TOKENS:
        # Calculate tokens per chunk to stay within limits
        tokens_per_chunk = min(TARGET_CHUNK_TOKENS, total_tokens // 3)
        
        # Initialize encoder once for efficiency
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(conversation_text)
        
        # Get chunk positions in tokens
        start_tokens = tokens[:tokens_per_chunk]
        mid_point = len(tokens) // 2
        middle_tokens = tokens[mid_point - tokens_per_chunk//2:mid_point + tokens_per_chunk//2]
        end_tokens = tokens[-tokens_per_chunk:]
        
        # Convert back to text
        start_chunk = encoder.decode(start_tokens)
        middle_chunk = encoder.decode(middle_tokens)
        end_chunk = encoder.decode(end_tokens)
        
        # Log chunk sizes for monitoring
        logger.debug(f"Chunk token counts - Start: {len(start_tokens)}, "
                    f"Middle: {len(middle_tokens)}, End: {len(end_tokens)}")
        
        prompt = (
            "This is a long conversation split into three parts. "
            "Analyze all parts and provide a detailed summary capturing the key points and progression of the conversation.\n"
            f"Beginning:\n{start_chunk}\n\n"
            f"Middle:\n{middle_chunk}\n\n"
            f"End:\n{end_chunk}"
        )
    else:
        prompt = conversation_text

    # Verify final prompt token count
    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > CLAUDE_MAX_TOKENS:
        logger.warning(f"Prompt too large ({prompt_tokens} tokens), truncating...")
        # Truncate to fit within limits if necessary
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(prompt)[:CLAUDE_MAX_TOKENS - 2000]  # Leave room for response
        prompt = encoder.decode(tokens)

    backoff = BACKOFF_INITIAL
    while True:
        try:
            async with semaphore:
                message = await client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=[{
                        "role": "user",
                        "content": (
                            "Analyze this conversation and return ONLY a single-line JSON object. "
                            "The response must be a JSON object with these fields:\n"
                            '{\n'
                            '  "title": "Descriptive title that captures the main topic",\n'
                            '  "summary": "Detailed summary (2-3 paragraphs) covering:\n'
                            '    - Main topic and context\n'
                            '    - Key points discussed\n'
                            '    - Important decisions or conclusions reached"\n'
                            '}\n\n'
                            "Important: Make the summary more detailed than usual as this is a longer conversation. "
                            "Ensure the entire response is on a single line with proper JSON escaping.\n"
                            "Do not include any other text or formatting.\n\n"
                            f"Conversation:\n{prompt}"
                        )
                    }]
                )
                
                # Log the raw response for debugging
                logger.debug(f"Raw API response: {message.content[0].text[:500]}...")
                
                try:
                    # Clean the response text
                    response_text = message.content[0].text.strip()
                    
                    # Try to find JSON object if there's other text
                    if not response_text.startswith('{'):
                        start = response_text.find('{')
                        end = response_text.rfind('}')
                        if start != -1 and end != -1:
                            response_text = response_text[start:end + 1]
                    
                    # Clean any trailing text after the JSON
                    if response_text.endswith('...'):
                        response_text = response_text[:response_text.rfind('}')+1]
                    
                    # Replace newlines and clean control characters in the JSON string itself
                    response_text = response_text.replace('\n', ' ').replace('\r', ' ')
                    response_text = ''.join(char if char.isprintable() or char.isspace() else ' ' 
                                          for char in response_text)
                    
                    # Parse the JSON
                    parsed = json.loads(response_text)
                    
                    # Validate and clean the fields
                    if not isinstance(parsed, dict):
                        raise ValueError("Response is not a dictionary")
                    if "title" not in parsed or "summary" not in parsed:
                        raise ValueError("Missing required fields")
                    if not isinstance(parsed["title"], str) or not isinstance(parsed["summary"], str):
                        raise ValueError("Fields are not strings")
                    
                    # Clean the strings
                    title = ' '.join(parsed["title"].split())  # Normalize whitespace
                    summary = ' '.join(parsed["summary"].split())  # Normalize whitespace
                    
                    # Remove any non-printable characters
                    title = ''.join(char for char in title if char.isprintable() or char.isspace())
                    summary = ''.join(char for char in summary if char.isprintable() or char.isspace())
                    
                    return {
                        "title": title.strip(),
                        "summary": summary.strip()
                    }
                    
                except (json.JSONDecodeError, ValueError, IndexError, KeyError) as e:
                    logger.error(f"Error parsing API response: {str(e)}")
                    logger.error(f"Raw response: {response_text[:500]}...")
                    return {"summary": "", "title": ""}
                
        except Exception as e:
            if "429" in str(e):
                wait_time = min(backoff, BACKOFF_MAX)
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry")
                await asyncio.sleep(wait_time)
                backoff *= 2
                continue
            logger.error(f"Error getting summary: {str(e)}")
            return {"summary": "", "title": ""}



async def process_file(filepath: str, client: AsyncAnthropic,
                      summary_collection: SummaryCollection,
                      semaphore: asyncio.Semaphore, 
                      tracker: ProcessingTracker) -> bool:
    if tracker.is_processed(filepath):
        logger.info(f"Skipping already processed file: {filepath}")
        return False
        
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Count tokens first
        token_count = count_tokens(content)
        # Calculate wait time based on token count (in seconds)
        wait_time = (token_count / TOKENS_PER_MINUTE) * 60
        logger.info(f"File has {token_count} tokens. Waiting {wait_time:.1f} seconds before next request")
        
        # Extract conversation content and limit its size
        conv_start = content.find("## Conversation")
        if conv_start == -1:
            logger.warning(f"No conversation section found in {filepath}")
            return False
            
        conversation_text = content[conv_start:conv_start + 8000]  # Limit size to avoid token issues
        
        summary_response = await get_summary(conversation_text, client, semaphore)
        if not summary_response.get("summary"):
            logger.warning(f"No valid summary generated for {filepath}")
            return False
            
        # Extract title and date
        title_match = content.find("# ")
        date_match = content.find("**Date:**")
        
        if title_match != -1 and date_match != -1:
            existing_title = content[title_match+2:content.find("\n", title_match)].strip()
            date_line = content[date_match:].split("\n")[0]
            date = date_line.replace("**Date:**", "").strip()
            
            if not summary_response["title"].strip():
                summary_response["title"] = existing_title
        
        # Update file
        date_end = content.find("**Token Count:**")
        if date_end == -1:
            logger.error(f"Could not find insertion point in {filepath}")
            return False
        
        new_content = (
            content[:date_end] +
            f"## Summary\n{summary_response['summary']}\n\n" +
            content[date_end:]
        )
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        # Update collections
        filename = os.path.basename(filepath)
        summary_collection.add_summary(
            filename, 
            summary_response["title"], 
            summary_response["summary"], 
            date
        )
        
        tracker.mark_processed(filepath, {
            "title": summary_response["title"],
            "summary": summary_response["summary"]
        })
        
        # Wait proportional to token count before next request
        await asyncio.sleep(wait_time)
        return True
            
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return False

async def main():
    # Update logging level to see more details
    logger.setLevel(logging.DEBUG)
    
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment variables")
        return
    
    client = AsyncAnthropic(api_key=api_key)
    summary_collection = SummaryCollection()
    tracker = ProcessingTracker(PROCESSED_FILE)
    markdown_dir = "/home/rationallyprime/Emma/markdown_exports/oversized"
    
    try:
        markdown_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(markdown_dir)
            for file in files
            if file.endswith('.md')
        ]
        
        if not markdown_files:
            logger.info("No oversized markdown files found to process")
            return
            
        logger.info(f"Found {len(markdown_files)} oversized markdown files to process")
        
        # Process in smaller batches with rate limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        for i in range(0, len(markdown_files), BATCH_SIZE):
            batch = markdown_files[i:i + BATCH_SIZE]
            results = await tqdm_asyncio.gather(
                *[process_file(f, client, summary_collection, semaphore, tracker) 
                  for f in batch],
                desc=f"Processing batch {i//BATCH_SIZE + 1}/{len(markdown_files)//BATCH_SIZE + 1}"
            )
            
            successful = sum(1 for r in results if r)
            logger.info(f"Batch {i//BATCH_SIZE + 1}: Successfully processed {successful}/{len(batch)} files")
            
            # Add longer delay between batches to respect rate limits
            await asyncio.sleep(2)

        logger.info(f"Summaries saved to {summary_collection.output_file}")
        
    except KeyboardInterrupt:
        logger.info("Processing paused by user. Progress saved.")

if __name__ == "__main__":
    asyncio.run(main())
