import json
from pathlib import Path
from datetime import datetime

def extract_conversations(json_file_path: str):
    # Debug: Print first 1000 chars of file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        print("\nFirst 1000 characters of file:")
        print(repr(content[:1000]))
        
        # Try parsing as JSON
        try:
            data = json.loads(content)
            print(f"\nSuccessfully parsed JSON, found {len(data)} conversations")
        except json.JSONDecodeError as e:
            print(f"\nError parsing JSON: {e}")
            print("Error location:", content[e.pos-50:e.pos+50])
            return
    
    # Define base output directory
    base_dir = Path('/home/rationallyprime/fellowship_data')
    
    for conv in data:
        # Get creation time and format folder name
        created_at = datetime.fromisoformat(conv.get('created_at').replace('Z', '+00:00'))
        folder_name = created_at.strftime("%Y-%m")
        
        # Create folder if it doesn't exist
        folder_path = base_dir / folder_name
        folder_path.mkdir(exist_ok=True, parents=True)
        
        title = conv.get('name', 'Untitled')
        messages = []
        
        # Get messages directly from chat_messages array and sort by timestamp
        chat_messages = conv.get('chat_messages', [])
        
        # Sort messages by created_at timestamp
        sorted_messages = sorted(chat_messages, key=lambda m: m.get('created_at', ''))
        
        for msg in sorted_messages:
            role = msg.get('sender', '').lower()
            content_list = msg.get('content', [])
            
            if role in ['human', 'assistant']:
                # Convert 'human' to 'user' for consistency
                display_role = 'user' if role == 'human' else role
                
                if role == 'assistant' and content_list and isinstance(content_list, list):
                    # Assistant messages may have thinking followed by text
                    has_thinking = False
                    
                    for item in content_list:
                        if item.get('type') == 'thinking':
                            # The thinking content is in the main message's text field
                            thinking_content = msg.get('text', '')
                            if thinking_content:
                                messages.append(('assistant_thinking', thinking_content))
                                has_thinking = True
                        elif item.get('type') == 'text' and item.get('text'):
                            # The actual response is in the content item
                            text_content = item.get('text', '')
                            messages.append((display_role, text_content))
                            
                elif role == 'human' and content_list and isinstance(content_list, list):
                    # Human messages have text in the content array
                    for item in content_list:
                        if item.get('type') == 'text' and item.get('text'):
                            text_content = item.get('text', '')
                            messages.append((display_role, text_content))
                            break
        
        if messages:
            # Create safe filename from title and date
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            filename = f"{created_at.strftime('%Y%m%d')}_{safe_title}.md"
            
            output_file = folder_path / filename
            write_to_markdown(messages, output_file)
            print(f"Wrote conversation to {output_file}")

def write_to_markdown(messages, output_file: Path):
    with open(output_file, 'w', encoding='utf-8') as f:
        for role, content in messages:
            if role == 'assistant_thinking':
                f.write(f"### Assistant (thinking)\n")
                f.write(f"<thinking>\n{content}\n</thinking>\n\n")
            else:
                f.write(f"### {role.capitalize()}\n")
                f.write(f"{content}\n\n")

# Example usage
json_file = 'conversations.json'
extract_conversations(json_file)