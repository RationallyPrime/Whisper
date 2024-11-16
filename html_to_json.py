from bs4 import BeautifulSoup

def extract_user_assistant_messages(html_file_path: str):
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Assuming user and assistant messages are within specific tags or classes
    # You may need to adjust the selectors based on the actual HTML structure
    messages = []
    for div in soup.find_all('div', class_='text-message'):
        role = div.get('data-message-author-role')
        if role in ['user', 'assistant']:
            message_content = div.get_text(strip=True)
            messages.append((role, message_content))

    return messages

# Example usage
html_file_path = 'logs/GPT_debate.html'
messages = extract_user_assistant_messages(html_file_path)

for role, content in messages:
    print(f"{role.capitalize()}: {content}\n")
