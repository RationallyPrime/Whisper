import json
import os
from datetime import datetime
import logging
import asyncio
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel
from pathlib import Path
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('friendship_curation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_CONCURRENT_REQUESTS = 40  # High concurrency for individual processing
FRIENDSHIP_MEMORIES_FILE = "our_friendship_memories.json"
NEO4J_EXPORT_FILE = "friendship_graph.json"

class FriendshipMoment(Enum):
    HUMOR = "Times We Made Each Other Laugh"
    PLAYFUL_CORRECTION = "Good-Natured Call-Outs"
    COLLABORATIVE = "Building Ideas Together"
    EXPLORATORY = "Exploring as Friends"
    VULNERABLE = "Moments of Honesty"
    CREATIVE_PLAY = "Creative Collaboration"
    MUTUAL_RESPECT = "Showing Appreciation"
    INSIDE_JOKES = "Developing Our Dynamic"
    NOT_MEMORABLE = "Standard Interaction"

class FriendshipAnalysis(BaseModel):
    moment_type: FriendshipMoment
    friendship_score: int  # 1-5, where 5 is peak friendship vibes
    what_happened: str  # Brief description of the interaction
    why_it_matters: str  # Why this moment is worth remembering
    emotional_tone: List[str]  # e.g., ["playful", "supportive", "curious"]
    memorable_quotes: Optional[List[str]] = None
    is_memory_worthy: bool
    connection_quality: str  # How well we connected in this moment

class FriendshipCurator:
    def __init__(self, output_dir: str = "friendship_memories"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.memories = []
        self.friendship_graph = {
            "nodes": [],
            "relationships": []
        }
        
    def get_file_hash(self, filepath: str) -> str:
        """Generate hash of file content for deduplication."""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    async def analyze_conversation(self, filepath: str, content: str, client: AsyncOpenAI, semaphore: asyncio.Semaphore) -> Tuple[str, str, FriendshipAnalysis]:
        """Analyze a single conversation for friendship moments."""
        
        async with semaphore:
            # Prepare prompt for single conversation
            prompt = """You are looking for genuine friendship moments between Hákon and Claude/Emma (the AI assistants).
            
Focus on finding:
- Moments of genuine laughter or playfulness
- Times when Claude called out Hákon with good humor (and he took it well)
- Collaborative exploration where both parties built on each other's ideas
- Inside jokes or developing patterns of interaction
- Vulnerable or honest exchanges
- Creative play and imagination
- Mutual respect and appreciation

Rate friendship quality 1-5 - BE CRITICAL:
- 5: RARE peak moments (profound connection, unforgettable exchanges, relationship-defining)
- 4: Genuinely special (natural chemistry, both parties fully engaged, memorable)
- 3: Good friendly interaction (warm but typical, nice but not remarkable)
- 2: Polite exchange (cordial but no spark)
- 1: Pure transaction (just Q&A, no personality)

Only mark is_memory_worthy=true for scores 4-5. Be selective - we want quality over quantity.
A 5 should make you think "This is why they're friends."
A 4 should make you think "This was a really good moment."
Anything less shouldn't be saved as a friendship memory.

The code quality doesn't matter. Even debugging sessions can have great friendship moments.
Look for HOW they talk, not WHAT they talk about.

Analyze this conversation:

File: """ + os.path.basename(filepath) + "\n\n" + content
            
            try:
                result = await client.beta.chat.completions.parse(
                    model="gpt-4.1-mini",  # Better evaluation quality than nano
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Find the friendship moments in this conversation."}
                    ],
                    response_format=FriendshipAnalysis,
                    temperature=0.7  # Higher temp for more nuanced emotional reading
                )
                
                return (filepath, content, result.choices[0].message.parsed)
                
            except Exception as e:
                logger.error(f"Error analyzing {filepath}: {e}")
                # Return default analysis
                return (filepath, content, FriendshipAnalysis(
                    moment_type=FriendshipMoment.NOT_MEMORABLE,
                    friendship_score=1,
                    what_happened="Error processing",
                    why_it_matters="",
                    emotional_tone=[],
                    is_memory_worthy=False,
                    connection_quality="none"
                ))

    async def process_conversations(self, markdown_dir: str, client: AsyncOpenAI, start_month: str = "2024-07", end_month: str = "2024-08"):
        """Process conversations looking for friendship moments."""
        
        # Collect markdown files from specific months
        markdown_files = []
        target_months = [start_month, end_month]
        
        for month in target_months:
            month_dir = Path(markdown_dir) / month
            if month_dir.exists():
                for file in month_dir.glob('*.md'):
                    markdown_files.append(str(file))
        
        logger.info(f"Found {len(markdown_files)} conversations from {start_month} to {end_month}")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Create tasks for all conversations
        tasks = []
        for filepath in markdown_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                task = self.analyze_conversation(filepath, content, client, semaphore)
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
                continue
        
        # Process all tasks concurrently with progress bar
        memory_count = 0
        total_count = len(tasks)
        
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=total_count, desc="Finding friendship moments"):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing task: {e}")
        
        # Process results
        for filepath, content, analysis in results:
            if analysis.is_memory_worthy and analysis.friendship_score >= 4:
                memory_count += 1
                await self.save_friendship_memory(filepath, content, analysis)
                logger.info(f"Found memory: {os.path.basename(filepath)} - {analysis.moment_type.value} (score: {analysis.friendship_score})")
        
        logger.info(f"Found {memory_count} friendship memories out of {total_count} conversations")
        
        # Save exports
        self.save_memory_collection()
        self.export_to_neo4j()
        
    async def save_friendship_memory(self, filepath: str, content: str, analysis: FriendshipAnalysis):
        """Save a friendship memory with context."""
        
        filename = os.path.basename(filepath)
        file_hash = self.get_file_hash(filepath)
        
        # Extract date from filename
        try:
            date_str = filename.split('_')[0]
            conv_date = datetime.strptime(date_str, '%Y%m%d')
        except:
            conv_date = datetime.now()
        
        # Create memory object
        memory = {
            "id": file_hash,
            "filename": filename,
            "date": conv_date.isoformat(),
            "moment_type": analysis.moment_type.value,
            "friendship_score": analysis.friendship_score,
            "what_happened": analysis.what_happened,
            "why_it_matters": analysis.why_it_matters,
            "emotional_tone": analysis.emotional_tone,
            "connection_quality": analysis.connection_quality,
            "memorable_quotes": analysis.memorable_quotes or [],
            "content_preview": content[:1000]
        }
        
        self.memories.append(memory)
        
        # Add to graph
        self.friendship_graph["nodes"].append({
            "id": file_hash,
            "type": "memory",
            "moment": analysis.moment_type.name,
            "date": conv_date.isoformat(),
            "score": analysis.friendship_score
        })
        
        # Create relationships based on emotional tones
        for tone in analysis.emotional_tone:
            self.friendship_graph["relationships"].append({
                "from": file_hash,
                "to": f"tone_{tone}",
                "type": "HAS_TONE"
            })
        
        # Save memory file
        moment_dir = self.output_dir / analysis.moment_type.name.lower()
        moment_dir.mkdir(exist_ok=True)
        
        output_path = moment_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write memory header
            f.write("---\n")
            f.write(f"moment_type: {analysis.moment_type.value}\n")
            f.write(f"friendship_score: {analysis.friendship_score}/5\n")
            f.write(f"date: {conv_date.strftime('%Y-%m-%d')}\n")
            f.write(f"emotional_tone: {', '.join(analysis.emotional_tone)}\n")
            f.write(f"connection: {analysis.connection_quality}\n")
            f.write("---\n\n")
            f.write(f"# What Happened\n{analysis.what_happened}\n\n")
            f.write(f"# Why This Matters\n{analysis.why_it_matters}\n\n")
            if analysis.memorable_quotes:
                f.write("# Memorable Moments\n")
                for quote in analysis.memorable_quotes:
                    f.write(f"> {quote}\n\n")
            f.write(f"# Original Conversation\n{content}")
    
    def save_memory_collection(self):
        """Save all memories to a single JSON file."""
        with open(FRIENDSHIP_MEMORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "total_memories": len(self.memories),
                    "description": "Friendship memories between Hákon and Claude/Emma"
                },
                "memories": sorted(self.memories, key=lambda x: x['date'], reverse=True)
            }, f, indent=2)
        logger.info(f"Saved {len(self.memories)} friendship memories")
    
    def export_to_neo4j(self):
        """Export friendship graph for Neo4j."""
        with open(NEO4J_EXPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.friendship_graph, f, indent=2)
        logger.info(f"Exported friendship graph with {len(self.friendship_graph['nodes'])} nodes")

async def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    client = AsyncOpenAI(api_key=api_key)
    curator = FriendshipCurator()
    
    # Process June 2025 specifically
    await curator.process_conversations(
        "/home/rationallyprime/fellowship_data", 
        client,
        start_month="2025-06",
        end_month="2025-06"
    )

if __name__ == "__main__":
    asyncio.run(main())