from pathlib import Path
import re
from datetime import datetime

def parse_chat_header(text: str) -> tuple[str, datetime]:
    """Extract title and date from chat header."""
    # Extract title
    title_match = re.match(r'# (.*?)\n', text)
    if not title_match:
        return None, None
    title = title_match.group(1)
    
    # Extract date
    date_match = re.match(r'# .*?\nDate: (.*?)\n', text, re.DOTALL)
    if not date_match:
        return None, None
    try:
        date = datetime.strptime(date_match.group(1), '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None, None
        
    return title, date

def sanitize_filename(title: str) -> str:
    """Convert title to safe filename."""
    # Replace invalid filename characters with underscores
    safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
    # Remove or replace other problematic characters
    safe_title = safe_title.replace('\n', '_').replace('\r', '_')
    # Limit length and trim spaces
    return safe_title[:100].strip()

def split_markdown_file(input_file: Path):
    """Split a markdown file into separate files for each conversation."""
    content = input_file.read_text(encoding='utf-8')
    
    # Split content by markdown separator
    conversations = content.split('\n\n---\n\n')
    
    # Create output directory based on input filename (e.g., chats_2024-01 -> 2024-01/)
    month_dir = input_file.parent / input_file.stem.replace('chats_', '')
    month_dir.mkdir(exist_ok=True)
    
    for conversation in conversations:
        # Skip empty conversations
        if not conversation.strip():
            continue
            
        # Parse header
        title, date = parse_chat_header(conversation)
        if not title or not date:
            print(f"Warning: Could not parse header in {input_file}")
            continue
            
        # Create filename: YYYY-MM-DD_Title.md
        safe_title = sanitize_filename(title)
        filename = f"{date.strftime('%Y-%m-%d')}_{safe_title}.md"
        output_file = month_dir / filename
        
        # Save conversation
        output_file.write_text(conversation, encoding='utf-8')
        print(f"Created {output_file}")

def main():
    input_dir = Path('/home/rationallyprime/formatted_chats')
    
    # Process each markdown file
    for input_file in input_dir.glob('chats_*.md'):
        split_markdown_file(input_file)

if __name__ == '__main__':
    main()

