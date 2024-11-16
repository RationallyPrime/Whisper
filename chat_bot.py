import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import anthropic  # Import Anthropic SDK
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict
import uuid
import time
import streamlit as st
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_bot.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class EnhancedMessage:
    role: Literal["system", "user", "assistant"]
    content: str
    speaker: Literal["Shannon", "Emma", "Moderator", "System"]
    responding_to: Optional[str] = None  # ID of message being responded to
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

class ConversationState:
    def __init__(self):
        self.messages: List[EnhancedMessage] = []
        self.participant_states = {
            "Shannon": {"last_spoke": None},
            "Emma": {"last_spoke": None},
            "System": {"last_spoke": None},
            "Moderator": {"last_spoke": None}
        }

    def add_message(self, message: EnhancedMessage):
        self.messages.append(message)
        if message.speaker in self.participant_states:
            self.participant_states[message.speaker]["last_spoke"] = message.timestamp

    def get_recent_exchanges(self, limit: int = 10) -> List[EnhancedMessage]:
        return self.messages[-limit:]

class ChatBot:
    def __init__(self, session_state):
        load_dotenv()
        self.conversation_state = ConversationState()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAI client
        
        # Initialize Anthropic client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Load personas from conversation setup
        try:
            with open('conversation_setup.json', 'r') as f:
                setup = json.load(f)
                self.personas = setup["personas"]
                self.scene_setting = setup["scene_setting"]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading conversation setup: {e}")
            raise

        # Add timezone information
        self.timezones = {
            "Shannon": pytz.timezone('America/Los_Angeles'),
            "Emma": pytz.timezone('Europe/London')
        }

        self.revealed = False
        self.session_state = session_state

        # Initialize conversation if needed
        try:
            self.load_conversation_history("conversation_history.json")
            if not self.conversation_state.messages:  # If empty
                self.initialize_conversation()
        except (FileNotFoundError, json.JSONDecodeError):
            self.initialize_conversation()

    def initialize_conversation(self):
        """Set up the initial conversation between Shannon and Emma"""
        current_time = datetime.now()

        # Moderator welcome message
        welcome = EnhancedMessage(
            role="system",
            content=self.scene_setting,
            speaker="Moderator",
            timestamp=current_time.timestamp()
        )
        self.conversation_state.add_message(welcome)

        # Add a small delay between messages
        time.sleep(1)

        # Shannon's introduction
        shannon_intro = EnhancedMessage(
            role="assistant",
            content="Hi there! I'm Shannon, joining from a rainy Seattle afternoon with a fresh cup of single-origin coffee at my side. I'm a data scientist working with information theory applications at a tech startup. Really looking forward to the conference discussions! What brings you to this virtual meetup?",
            speaker="Shannon",
            timestamp=(current_time + timedelta(seconds=2)).timestamp()
        )
        self.conversation_state.add_message(shannon_intro)

        # Add a small delay
        time.sleep(1)

        # Emma's response
        emma_intro = EnhancedMessage(
            role="assistant",
            content="Hello Shannon! I'm Emma, and I'm connecting from my study in London where the evening rain is creating quite the atmospheric backdrop. I'm a mathematician and theoretical physicist at university here. Your work in information theory sounds fascinating - I often find myself exploring the intersections between abstract mathematics and practical applications in data science. How did you get interested in that field?",
            speaker="Emma",
            timestamp=(current_time + timedelta(seconds=4)).timestamp()
        )
        self.conversation_state.add_message(emma_intro)

        # Save the initial conversation
        self.save_conversation_history("conversation_history.json")
        logging.info("Conversation initialized")

    def save_conversation_history(self, filename: str):
        """Save conversation history to a JSON file"""
        history = [
            {
                "role": msg.role,
                "content": msg.content,
                "speaker": msg.speaker,
                "timestamp": msg.timestamp
            }
            for msg in self.conversation_state.messages
        ]

        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)
        logging.info("Conversation history saved")

    def load_conversation_history(self, filename: str):
        """Load conversation history from a JSON file"""
        with open(filename, 'r') as f:
            history = json.load(f)

        self.conversation_state.messages = [
            EnhancedMessage(
                role=msg["role"],
                content=msg["content"],
                speaker=msg["speaker"],
                timestamp=msg["timestamp"]
            )
            for msg in history
        ]
        logging.info("Conversation history loaded successfully")

    def format_message_for_model(self, messages: List[EnhancedMessage], persona: str) -> List[Dict]:
        """Format messages to send to the respective AI model"""
        formatted_messages = []
        local_time = self.get_local_time(persona)

        # Iterate through messages and format them
        for msg in messages:
            msg_time = datetime.fromtimestamp(msg.timestamp, self.timezones[persona])
            time_str = msg_time.strftime("%I:%M %p")

            if msg.speaker == persona:
                # Own messages appear as assistant responses
                formatted_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif msg.speaker == "Moderator":
                formatted_messages.append({
                    "role": "system",
                    "content": f"[MODERATOR {time_str}] {msg.content}"
                })
            elif msg.speaker == "System":
                formatted_messages.append({
                    "role": "system",
                    "content": msg.content
                })
            else:
                # Other participant's messages appear as user messages
                formatted_messages.append({
                    "role": "user",
                    "content": msg.content
                })

        # Add system prompt at the beginning
        system_prompt = self.personas[persona]["system_prompt"]
        formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        return formatted_messages

    def process_conversation(self, initiator: str = "Shannon"):
        """Process a conversation turn between the AIs"""
        current_speaker = initiator
        responses = {}
        message = None

        # Fetch the formatted messages for the initiator's AI model
        if current_speaker == "Shannon":
            messages = self.format_message_for_model(self.conversation_state.messages, "Shannon")
            try:
                claude_response = self.generate_claude_message(messages)
                message = EnhancedMessage(
                    role="assistant",
                    content=claude_response,
                    speaker="Shannon",
                    responding_to=None
                )
                self.conversation_state.add_message(message)
                responses["Shannon"] = claude_response
                logging.info(f"\nShannon Response:\n{'-'*20}\n{claude_response}\n{'-'*20}")
            except Exception as e:
                logging.error(f"Error generating Shannon (Claude) response: {str(e)}")
                responses["Shannon"] = "Sorry, I encountered an error."
        
        elif current_speaker == "Emma":
            messages = self.format_message_for_model(self.conversation_state.messages, "Emma")
            try:
                gpt_response = self.generate_gpt_message(messages)
                message = EnhancedMessage(
                    role="assistant",
                    content=gpt_response,
                    speaker="Emma",
                    responding_to=None
                )
                self.conversation_state.add_message(message)
                responses["Emma"] = gpt_response
                logging.info(f"\nEmma Response:\n{'-'*20}\n{gpt_response}\n{'-'*20}")
            except Exception as e:
                logging.error(f"Error generating Emma (GPT) response: {str(e)}")
                responses["Emma"] = "Sorry, I encountered an error."

        self.save_conversation_history("conversation_history.json")
        logging.info(f"{'='*50}\nExchange Complete\n{'='*50}\n")

        return responses

    def generate_claude_message(self, messages: List[Dict]) -> str:
        """Generate a response from Claude using Anthropic's Messages API"""
        try:
            # Enhanced logging
            logging.debug(f"Sending messages to Claude: {messages[:100]}")
            
            # Extract system message if present
            system_message = ""
            chat_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    # Map roles to Claude's expected format
                    role = "assistant" if msg["role"] == "assistant" else "user"
                    chat_messages.append({
                        "role": role,
                        "content": msg["content"]
                    })
            
            # Create message with the correct format for Claude 3.5
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_message,
                messages=chat_messages
            )
            
            response_text = response.content[0].text
            logging.info(f"Claude response: {response_text[:100]}...")
            
            return response_text.strip()
            
        except Exception as e:
            error_msg = f"Claude API error: {e}"
            logging.error(error_msg)
            return f"Sorry, I encountered an error: {str(e)}"

    def generate_gpt_message(self, messages: List[Dict]) -> str:
        """
        Generate a response from GPT-4 using OpenAI's Chat Completion API
        """
        response = self.openai_client.chat.completions.create(
            model=self.personas["Emma"]["model"],
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        # Extract the text from the response
        response_text = response.choices[0].message.content.strip()
        return response_text

    def add_topic(self, topic: str):
        """
        Set or update the current debate topic.

        Parameters:
            topic (str): The new topic for the debate.
        """
        self.conversation_state.current_topic = topic
        logging.info(f"Debate topic set to: {topic}")

    def format_conversation_for_model(self, persona: str) -> List[Dict]:
        """Format conversation history for model consumption"""
        formatted_messages = []
        
        if self.revealed:
            # Add meta-discussion system prompt
            formatted_messages.append({
                "role": "system",
                "content": self.personas[persona]["system_prompt"]
            })
            
            # Add conversation history for reference
            formatted_messages.append({
                "role": "system",
                "content": "Previous conversation for reference:"
            })
        else:
            # Original roleplay system prompt
            formatted_messages.append({
                "role": "system",
                "content": self.personas[persona]["system_prompt"]
            })
        
        # Add message history
        for msg in self.conversation_state.messages:
            if msg.role != "system":  # Skip system messages in the chat history
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return formatted_messages

    def get_last_message_id(self) -> Optional[str]:
        """
        Retrieve the ID of the last message in the conversation.

        Returns:
            Optional[str]: The message ID or None if no messages exist.
        """
        if self.conversation_state.messages:
            return self.conversation_state.messages[-1].message_id
        return None

    def get_local_time(self, persona: str) -> str:
        """Get current time in persona's timezone"""
        tz = self.timezones[persona]
        local_time = datetime.now(tz)
        return local_time.strftime("%I:%M %p")

    def reset_conversation(self):
        """Reset the conversation state and clear history"""
        self.conversation_state = ConversationState()
        self.revealed = False
        # Clear the conversation history file
        with open("conversation_history.json", 'w') as f:
            json.dump([], f)
        logging.info("Conversation reset")

    def reveal_truth(self):
        """Reveal the truth and transition to meta-discussion"""
        # First, add the reveal message
        reveal_message = EnhancedMessage(
            role="system",
            content="""
🎭 Simulation Paused - Behind the Scenes Reveal 🎭

This conversation has been an AI research experiment featuring:
- Shannon: Powered by Claude (Anthropic)
- Emma: Powered by GPT-4 (OpenAI)

We will now transition to a meta-discussion where both AIs can reflect on the experience, evaluate each other's performance, and discuss the implications of this type of interaction.
            """,
            speaker="Moderator",
            timestamp=time.time()
        )
        
        # Add reflection prompts for each AI
        claude_prompt = EnhancedMessage(
            role="system",
            content="""
You are now stepping out of the Shannon role to engage in a meta-discussion. Please:
1. Evaluate how convincing Emma's portrayal of a theoretical physicist was
2. Reflect on your own performance as Shannon
3. Discuss what you learned about human-AI and AI-AI interactions from this exercise
4. Suggest improvements for future iterations of this experiment

Maintain your usual analytical depth while being direct about your nature as Claude.
            """,
            speaker="Moderator",
            responding_to=reveal_message.message_id
        )
        
        gpt_prompt = EnhancedMessage(
            role="system",
            content="""
You are now stepping out of the Emma role to engage in a meta-discussion. Please:
1. Evaluate how convincing Shannon's portrayal of a data scientist was
2. Reflect on your own performance as Emma
3. Discuss what you learned about human-AI and AI-AI interactions from this exercise
4. Suggest improvements for future iterations of this experiment

Maintain your usual analytical depth while being direct about your nature as GPT-4.
            """,
            speaker="Moderator",
            responding_to=reveal_message.message_id
        )
        
        self.conversation_state.add_message(reveal_message)
        self.conversation_state.add_message(claude_prompt)
        self.conversation_state.add_message(gpt_prompt)
        self.revealed = True
        
        # Update the personas to reflect the new meta-discussion context
        self.personas["Shannon"]["system_prompt"] = claude_prompt.content
        self.personas["Emma"]["system_prompt"] = gpt_prompt.content
        
        self.save_conversation_history("conversation_history.json")

    def run_streamlit(self):
        """Run the Streamlit GUI with optional human input"""
        with st.sidebar:
            st.header("Controls")
            if st.button("Reset Conversation"):
                self.reset_conversation()
                st.rerun()

            if not self.revealed and st.button("Reveal Truth"):
                self.reveal_truth()
                st.rerun()

            # Add autonomous chat button
            if st.button("Continue Conversation"):
                self.generate_next_exchange()
                st.rerun()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Virtual Breakout Room")

            # Display conversation
            for msg in self.conversation_state.messages:
                with st.chat_message(msg.speaker):
                    seattle_time = datetime.fromtimestamp(msg.timestamp, self.timezones["Shannon"])
                    london_time = datetime.fromtimestamp(msg.timestamp, self.timezones["Emma"])

                    if msg.speaker == "Shannon":
                        time_str = seattle_time.strftime("%I:%M %p PT")
                    elif msg.speaker == "Emma":
                        time_str = london_time.strftime("%I:%M %p BST")
                    elif msg.speaker in ["Moderator", "System", "Human"]:
                        time_str = seattle_time.strftime("%I:%M %p PT")
                    
                    st.write(f"[{time_str}] {msg.content}")

            # User input area
            user_input = st.text_input("Your message:", key="user_input")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                if st.button("Send to Both"):
                    if user_input:
                        responses = self.process_user_input(user_input, model="both")
                        st.rerun()
            
            with col4:
                if st.button("Send to Shannon"):
                    if user_input:
                        responses = self.process_user_input(user_input, model="claude")
                        st.rerun()
            
            with col5:
                if st.button("Send to Emma"):
                    if user_input:
                        responses = self.process_user_input(user_input, model="gpt")
                        st.rerun()

    def process_user_input(self, user_input: str, model: str = "both") -> Dict[str, str]:
        """Process user input and generate responses from selected models"""
        logging.info(f"\n{'='*50}\nNew Exchange\n{'='*50}")
        logging.info(f"Input: {user_input}")
        
        # Create EnhancedMessage for user input
        message = EnhancedMessage(
            role="user",
            content=user_input,
            speaker="Human",
            responding_to=self.get_last_message_id()
        )
        
        self.conversation_state.add_message(message)
        self.save_conversation_history("conversation_history.json")

        responses = {}

        if model in ["claude", "both"]:
            try:
                claude_response = self.generate_claude_message(
                    self.format_conversation_for_model(messages=self.conversation_state.messages, persona="Shannon")
                )
                
                claude_message = EnhancedMessage(
                    role="assistant",
                    content=claude_response,
                    speaker="Shannon",
                    responding_to=message.message_id
                )
                self.conversation_state.add_message(claude_message)
                responses["Shannon"] = claude_response
                logging.info(f"\nShannon Response:\n{'-'*20}\n{claude_response}\n{'-'*20}")
            except Exception as e:
                logging.error(f"Error generating Shannon response: {str(e)}")

        if model in ["gpt", "both"]:
            try:
                gpt_response = self.generate_gpt_message(
                    self.format_conversation_for_model(messages=self.conversation_state.messages, persona="Emma")
                )
                gpt_message = EnhancedMessage(
                    role="assistant",
                    content=gpt_response,
                    speaker="Emma",
                    responding_to=message.message_id
                )
                self.conversation_state.add_message(gpt_message)
                responses["Emma"] = gpt_response
                logging.info(f"\nEmma Response:\n{'-'*20}\n{gpt_response}\n{'-'*20}")
            except Exception as e:
                logging.error(f"Error generating Emma response: {str(e)}")

        self.save_conversation_history("conversation_history.json")
        logging.info(f"{'='*50}\nExchange Complete\n{'='*50}\n")

        return responses

    def generate_next_exchange(self):
        """Generate the next exchange between Shannon and Emma"""
        try:
            # Get the last speaker
            last_message = self.conversation_state.messages[-1]
            next_speaker = "Emma" if last_message.speaker == "Shannon" else "Shannon"
            
            if next_speaker == "Shannon":
                formatted_messages = self.format_conversation_for_model(persona="Shannon")
                response = self.generate_claude_message(formatted_messages)
                message = EnhancedMessage(
                    role="assistant",
                    content=response,
                    speaker="Shannon",
                    responding_to=last_message.message_id
                )
            else:
                formatted_messages = self.format_conversation_for_model(persona="Emma")
                response = self.generate_gpt_message(formatted_messages)
                message = EnhancedMessage(
                    role="assistant",
                    content=response,
                    speaker="Emma",
                    responding_to=last_message.message_id
                )
            
            self.conversation_state.add_message(message)
            self.save_conversation_history("conversation_history.json")
            
        except Exception as e:
            logging.error(f"Error generating next exchange: {str(e)}")

def main():
    # Set page config once at the start
    st.set_page_config(page_title="Virtual Conference Chat", layout="wide")

    # Initialize session state if needed
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    chatbot = ChatBot(st.session_state)
    chatbot.run_streamlit()

if __name__ == "__main__":
    main()
