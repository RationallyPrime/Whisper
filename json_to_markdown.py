import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

def parse_timestamp(ts_str: str) -> datetime:
    """Convert ISO timestamp string to datetime object."""
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

def format_chat(chat_data: Dict) -> tuple[datetime, str]:
    """Convert a chat JSON into markdown format. Returns (first_message_time, formatted_text)."""
    
    # Extract chat info
    chat_title = chat_data.get('name', 'Untitled Chat')
    messages = chat_data.get('chat_messages', [])
    
    # Skip if no messages
    if not messages:
        return None
    
    # Sort messages by timestamp
    sorted_messages = sorted(messages, key=lambda x: parse_timestamp(x['created_at']))
    first_message_time = parse_timestamp(sorted_messages[0]['created_at'])
    
    # Format output
    output = [
        f"# {chat_title}",
        f"Date: {first_message_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    ]
    
    # Process messages in chronological order
    for msg in sorted_messages:
        sender = msg['sender'].upper()
        text = msg['text'].strip()
        timestamp = parse_timestamp(msg['created_at'])
        
        if text:  # Skip empty messages
            output.append(f"**{sender}** [{timestamp.strftime('%H:%M:%S')}]: {text}\n")
    
    return first_message_time, '\n'.join(output)

def process_chats(json_data: List[Dict]) -> Dict[str, List[tuple[datetime, str]]]:
    """Process chats and organize by month. Returns {month_key: [(timestamp, formatted_text)]}."""
    chats_by_month = {}
    
    # Filter for October 2024 chats only
    october_chats = [
        chat for chat in json_data 
        if parse_timestamp(chat['created_at']).strftime('%Y-%m') == '2024-10'
    ]
    
    for chat in october_chats:
        formatted = format_chat(chat)
        if formatted is None:  # Skip chats with no messages
            continue
            
        month_key = formatted[0].strftime('%Y-%m')
        
        if month_key not in chats_by_month:
            chats_by_month[month_key] = []
        chats_by_month[month_key].append(formatted)
    
    # Sort chats within each month by timestamp (descending)
    for month in chats_by_month:
        chats_by_month[month].sort(key=lambda x: x[0], reverse=True)
    
    return chats_by_month

def save_markdown_files(chats_by_month: Dict[str, List[tuple[datetime, str]]]):
    """Save formatted chats to markdown files organized by month."""
    output_dir = Path('/home/rationallyprime/formatted_chats')
    output_dir.mkdir(exist_ok=True)
    
    for month, chats in chats_by_month.items():
        output_text = '\n\n---\n\n'.join(chat[1] for chat in chats)
        
        output_file = output_dir / f'chats_{month}.md'
        output_file.write_text(output_text, encoding='utf-8')
        print(f"Created {output_file}")

def main():
    # Read input JSON
    with open('conversations.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process and save chats
    chats_by_month = process_chats(data)
    save_markdown_files(chats_by_month)

if __name__ == '__main__':
    main()
