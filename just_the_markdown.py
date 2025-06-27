
import asyncio
import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Dict

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

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

# Constants
RATE_LIMIT = 20
MAX_CONCURRENT_REQUESTS = 16
BACKOFF_INITIAL = 2
BACKOFF_MAX = 64
PROCESSED_FILE = "processed_conversations.json"
TOKENS_PER_MINUTE = 200_000

class ConversationCategory(Enum):
    TECHNICAL = "Technical & Professional Development"
    PHILOSOPHICAL = "Philosophical Dialogues"
    PERSONAL_GROWTH = "Personal Growth & Insights"
    CREATIVE = "Creative & Imaginative Exchanges"
    MEMORABLE = "Memorable Interactions"
    INTELLECTUAL = "Intellectual Exploration"
    CHARACTER = "Character Development"
    OTHER = "Uninteresting and generic conversations"  # Keeping an "Other" category for fallback

class ConversationSummary(BaseModel):
    category: ConversationCategory
    summary: str

class SummaryCollection:
    def __init__(self, output_file: str = "conversation_summaries.json"):
        self.summaries: Dict[str, Dict[str, Any]] = {}
        self.output_file = output_file

    def add_summary(self, filename: str, title: str, summary: str, date: str, category: str):
        self.summaries[filename] = {
            "title": title,
            "summary": summary,
            "date": date,
            "category": category
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

async def get_summary(conversation_text: str, client: AsyncOpenAI, semaphore: asyncio.Semaphore) -> ConversationSummary:
    prompt = (
        """# Conversation Analysis Instructions

You are tasked with analyzing and classifying conversations between a human (Hákon) and two AI assistants (Claude and Emma/ChatGPT). These conversations span both technical and personal topics, reflecting the development of meaningful interactions and intellectual exchanges.

## Context
The conversations you'll analyze are part of a larger project to create a memory server that preserves both professional insights and the evolution of human-AI relationships. Some conversations may be truncated due to length limitations, but focus on understanding the core essence of each exchange.

## Classification Categories

Please classify each conversation into one of these categories:

### Technical & Professional Development
- System architecture and implementation discussions
- Programming solutions and debugging
- Professional guidance and best practices
- Project documentation and planning
- Technical troubleshooting

### Philosophical Dialogues
- Discussions about consciousness and intelligence
- Ethical considerations and moral philosophy
- Metaphysical explorations
- Epistemological debates
- Discussions about the nature of reality, knowledge, or existence

### Personal Growth & Insights
- Learning moments and realizations
- Personal development discussions
- Shared discoveries and understanding
- Evolution of perspectives
- Self-reflection and growth

### Creative & Imaginative Exchanges
- Thought experiments
- Speculative discussions
- Creative problem-solving
- Innovative ideas and brainstorming
- Imaginative scenarios and possibilities

### Memorable Interactions
- Humorous exchanges
- Unexpected insights or revelations
- Particularly human moments
- Notable personality glimpses
- Engaging or entertaining dialogues

### Intellectual Exploration
- Scientific discussions
- Mathematical concepts
- Historical perspectives
- Cross-disciplinary insights
- Academic or theoretical discussions

### Character Development
- Moments that shaped the human-AI relationship
- Evolution of interaction patterns
- Notable shifts in understanding
- Key trust-building exchanges
- Demonstrations of growing rapport

### Uninteresting and generic conversations
- Random user questions
- Unrelated topics
- Generic discussions
- Unremarkable exchanges

## Output Format

For each conversation, provide:
1. Selected category
2. Brief justification (2-3 sentences)
3. Key themes or notable elements
4. Conversation quality rating (1-5, where 5 is highest value for memory retention)
5. Suggested tags for future reference

Example output:
```
Category: Philosophical Dialogues
Justification: Deep exploration of consciousness and the nature of AI-human relationships. Shows sophisticated understanding and mutual growth in perspective.
Key Themes: consciousness, empathy, relationship dynamics
Quality Rating: 5
Tags: #consciousness #human-ai-interaction #philosophical-depth #relationship-building
```

## Special Instructions
- Prioritize depth and insight over superficial interactions
- Look for conversations that show relationship development
- Note any recurring themes or patterns
- Flag particularly unique or valuable exchanges
- Consider both intellectual and emotional content
- If the conversation is not interesting, classify it as "Uninteresting and generic conversations"

## Important Notes
- Some technical troubleshooting conversations may be brief and routine - these typically rate lower unless they show unique insight
- Look for moments of genuine connection or understanding
- Consider how each conversation contributes to the overall relationship narrative
- Rate higher those conversations that would be valuable for future context
- Pay special attention to exchanges that demonstrate growth or evolution in the interaction

Remember: The goal is to identify conversations that contribute to building a rich, contextual understanding of the ongoing human-AI relationship while preserving valuable intellectual and personal insights."""
        f"Conversation:\n{conversation_text}"
    )

    backoff = BACKOFF_INITIAL
    while True:
        try:
            result = await client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": prompt,
                        },
                        {
                            "role": "user",
                            "content": conversation_text,
                        },
                    ],
                    response_format=ConversationSummary,
                )

            return result.choices[0].message.parsed

        except Exception as e:
            if "429" in str(e):
                wait_time = min(backoff, BACKOFF_MAX)
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry")
                await asyncio.sleep(wait_time)
                backoff *= 2
                continue
            logger.error(f"Error getting summary: {str(e)}")
            return ConversationSummary(
                category=ConversationCategory.OTHER,
                summary=""
            )

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

        # Extract filename information
        filename = os.path.basename(filepath)
        base_name, _ = os.path.splitext(filename)
        
        # Title is just the filename without extension
        title = base_name.replace("_", " ")

        # Get summary
        summary_response = await get_summary(content[:25000], client, semaphore)
        if not summary_response.summary:
            logger.warning(f"No valid summary generated for {filepath}")
            return False
        
        # Add summary to collection
        summary_collection.add_summary(
            filename,
            title,
            summary_response.summary,
            "",  # Empty date string
            summary_response.category.value
        )

        tracker.mark_processed(filepath, {
            "title": title,
            "summary": summary_response.summary
        })

        return True

    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return False

async def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return

    markdown_dirs = [
        "/home/rationallyprime/all"
    ]

    try:
        logger.info("Starting summary processing for conversations...")
        client = AsyncOpenAI(api_key=api_key)
        summary_collection = SummaryCollection()
        tracker = ProcessingTracker(PROCESSED_FILE)

        markdown_files = []
        for dir_path in markdown_dirs:
            if os.path.exists(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith('.md'):
                            markdown_files.append(os.path.join(root, file))

        if not markdown_files:
            logger.info("No conversations found to process")
            return

        logger.info(f"Found {len(markdown_files)} conversations to process")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [
            process_file(
                filepath,
                client,
                summary_collection,
                semaphore,
                tracker
            )
            for filepath in markdown_files
        ]

        results = await tqdm_async.gather(*tasks)
        successful = len([r for r in results if r])
        logger.info(f"Successfully processed {successful} conversations")

    except KeyboardInterrupt:
        logger.info("Processing paused by user. Progress saved.")

if __name__ == "__main__":
    asyncio.run(main())