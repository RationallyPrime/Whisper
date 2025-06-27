import json
from pathlib import Path
from datetime import datetime

def process_conversations(json_file: str, base_dir: Path):
    print(f"Processing {json_file}...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"\nSuccessfully loaded JSON, found {len(data)} conversations")
        
        for conv in data:
            messages = []
            mapping = conv.get('mapping', {})
            
            # Get conversation timestamps
            create_time = conv.get('create_time')
            if not create_time:
                print(f"Warning: No create_time for conversation '{conv.get('title')}', skipping...")
                continue
                
            date = datetime.fromtimestamp(create_time)
            folder_name = date.strftime("%Y-%m")  # e.g., "2024-03"
            
            # Find root message and follow the chain
            root_id = next(
                (msg_id for msg_id, msg_data in mapping.items() 
                 if msg_data.get('parent') is None),
                None
            )
            
            current_id = root_id
            while current_id and current_id in mapping:
                msg_data = mapping[current_id]
                if msg_data.get('message'):
                    msg = msg_data['message']
                    role = msg.get('author', {}).get('role', '').lower()
                    if role in ['user', 'assistant', 'tool']:
                        content = msg.get('content', {}).get('parts', [''])[0]
                        timestamp = msg.get('create_time')
                        if timestamp:
                            timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        messages.append((role, content, timestamp))
                
                # Move to next message
                children = msg_data.get('children', [])
                current_id = children[0] if children else None
            
            if messages:
                # Save conversation to file
                title = conv.get('title', 'Untitled')
                
                # Create month folder
                folder_path = base_dir / folder_name
                folder_path.mkdir(exist_ok=True, parents=True)
                
                # Create safe filename with timestamp
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title.replace(' ', '_')
                filename = f"{date.strftime('%Y%m%d_%H%M%S')}_{safe_title}.md"
                
                output_file = folder_path / filename
                write_to_markdown(messages, output_file, date)
                print(f"Wrote conversation to {output_file}")

def write_to_markdown(messages, output_file: Path, conv_date: datetime):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write conversation metadata
        f.write(f"# {output_file.stem}\n\n")
        f.write(f"Conversation started: {conv_date.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Write messages
        for role, content, timestamp in messages:
            f.write(f"### {role.capitalize()}")
            if timestamp:
                f.write(f" ({timestamp})")
            f.write("\n")
            f.write(f"{content}\n\n")

# Process the JSON file directly
json_file = 'conversations.json'
base_dir = Path('/home/rationallyprime/fellowship_data')
process_conversations(json_file, base_dir)