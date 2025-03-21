from typing import Dict
import re
import openai
import os
import json
from pathlib import Path

# Function to get API key
def get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
        
    config_path = Path.home() / '.npc_config.json'
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'api_key' in config:
                    use_saved = input("Found saved API key. Use it? (y/n): ").lower()
                    if use_saved == 'y':
                        return config['api_key']
        except:
            pass
    
    # Ask for API key - simple input for testing
    print("\nOpenAI API Key required for GPT-4 access.")
    api_key = input("Enter your OpenAI API key: ")
    
    return api_key


# Define the Personality class
class Personality:
    def __init__(self, openness: int, conscientiousness: int, agreeableness: int, neuroticism: int, extraversion: int):
        self.openness = openness
        self.conscientiousness = conscientiousness
        self.agreeableness = agreeableness
        self.neuroticism = neuroticism
        self.extraversion = extraversion

    def __str__(self) -> str:
        return (f"Openness: {self.openness}, Conscientiousness: {self.conscientiousness}, "
                f"Agreeableness: {self.agreeableness}, Neuroticism: {self.neuroticism}, "
                f"Extraversion: {self.extraversion}")


def fetch_gpt4_response(prompt: str, model_name: str = "gpt-4") -> str:
    """
    Use OpenAI's GPT-4 to generate a response for the given prompt.
    """
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an NPC in a game. Respond in character based on your personality traits and the scene description."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_personality_profile(world_description: str, npc_role: str) -> str:
    """
    Generate the prompt for GPT-4 to create NPC personality.
    """
    prompt = f"""
    Create an NPC for this fictional world: {world_description}
    The NPC's role is: {npc_role}
    
    Generate personality traits for this character on a scale from 0 to 100:
    Openness: (creativity, curiosity) 
    Conscientiousness: (organization, discipline)
    Agreeableness: (friendliness, cooperation)
    Neuroticism: (anxiety, emotional instability)
    Extraversion: (sociability, assertiveness)
    
    Also include a brief description of the NPC's behavior and mannerisms.
    Format your response so that each trait has a number followed by a brief explanation.
    """
    return fetch_gpt4_response(prompt)


def parse_personality_from_response(response: str) -> Personality:
    """
    Extracts personality traits from the GPT-4 response string.
    If traits aren't found, uses default values.
    """
    # Default values in case parsing fails
    traits = {
        "openness": 50,
        "conscientiousness": 50,
        "agreeableness": 50,
        "neuroticism": 50,
        "extraversion": 50
    }
    
    for trait in traits:
        match = re.search(fr"{trait.capitalize()}: (\d+)", response, re.IGNORECASE)
        if match:
            try:
                traits[trait] = int(match.group(1))
            except (ValueError, IndexError):
                pass
    
    print(f"Parsed personality traits: {traits}")
    print(f"From response: {response[:100]}...")
    
    return Personality(
        openness=traits["openness"],
        conscientiousness=traits["conscientiousness"],
        agreeableness=traits["agreeableness"],
        neuroticism=traits["neuroticism"],
        extraversion=traits["extraversion"]
    )

def dynamic_interaction(
        npc_personality: Personality, scene_description: str, conversation_history: str, player_input: str, model_name: str = "gpt-4"
) -> str:
    """
    Use OpenAI's GPT-4 to dynamically interact with the NPC based on personality, scene, and history.
    """
    if len(conversation_history) > 2000:
        conversation_history = conversation_history[-2000:]
    
    prompt = f"""
    Scene: {scene_description}
    
    NPC Personality:
    Openness: {npc_personality.openness}/100 (Higher means more creative, curious)
    Conscientiousness: {npc_personality.conscientiousness}/100 (Higher means more organized, disciplined)
    Agreeableness: {npc_personality.agreeableness}/100 (Higher means more friendly, cooperative)
    Neuroticism: {npc_personality.neuroticism}/100 (Higher means more anxious, emotionally volatile)
    Extraversion: {npc_personality.extraversion}/100 (Higher means more sociable, outgoing)
    
    Conversation history:
    {conversation_history}
    
    Player says: "{player_input}"
    
    Respond as this NPC would, keeping responses concise (1-3 sentences). Stay in character based on personality traits. Do not include "NPC:" before your response.
    """
    
    try:
        return fetch_gpt4_response(prompt, model_name=model_name)
    except Exception as e:
        return f"Error: {str(e)}"

def create_custom_personality() -> Personality:
    """
    Let the user create a custom NPC personality by entering trait values.
    """
    print("\nCreate your custom NPC:")
    print("Rate each trait on a scale of 0-100:")
    
    try:
        openness = int(input("Openness (creativity, curiosity): "))
        conscientiousness = int(input("Conscientiousness (organization, discipline): "))
        agreeableness = int(input("Agreeableness (friendliness, cooperation): "))
        neuroticism = int(input("Neuroticism (anxiety, emotional instability): "))
        extraversion = int(input("Extraversion (sociability, assertiveness): "))
        
        # Ensure values are within range
        traits = [openness, conscientiousness, agreeableness, neuroticism, extraversion]
        traits = [max(0, min(100, t)) for t in traits]  # Clamp between 0-100
        
        return Personality(
            openness=traits[0],
            conscientiousness=traits[1],
            agreeableness=traits[2],
            neuroticism=traits[3],
            extraversion=traits[4]
        )
    except ValueError:
        print("Invalid input. Using default values.")
        return Personality(50, 50, 50, 50, 50)


def npc_personality_system() -> None:
    """Main system for interacting with NPCs"""
    print("\n--- NPC Personality System ---")
    print("1. Use predefined NPCs")
    print("2. Create a custom NPC")
    choice = input("Enter your choice (1-2): ")
    
    # Default world setting
    world_description = "A medieval fantasy world with kingdoms and magic."
    
    if choice == "2":
        # Custom NPC and world
        custom_world = input("Describe the world (or press Enter for default): ")
        if custom_world.strip():
            world_description = custom_world
            
        npc_role = input("What is your NPC's role? (e.g., blacksmith, merchant): ")
        npc_personality = create_custom_personality()
        
    else:
        # Predefined NPCs
        print("\nChoose an NPC to talk to:")
        print("1. Blacksmith (outgoing, creative)")
        print("2. Merchant (reserved, disciplined)")
        npc_choice = input("Enter 1 or 2: ")
        
        if npc_choice == "1":
            npc_role = "blacksmith"
            npc_personality = Personality(10, 10, 10, 90, 90)
        else:
            npc_role = "merchant"
            npc_personality = Personality(90, 90, 90, 10, 10)
    
    print(f"\nEntering a conversation with the {npc_role}.")
    print(f"World: {world_description}")
    print(f"Personality: {npc_personality}")
    print("Type 'exit' to leave the conversation.")
    
    conversation_history = ""
    while True:
        player_input = input("\nYou: ")
        if player_input.lower() == 'exit':
            print("You left the conversation.")
            break
        response = dynamic_interaction(npc_personality, world_description, conversation_history, player_input)
        print(f"NPC: {response}")
        conversation_history += f"\nYou: {player_input}\nNPC: {response}"


# Entry point - this is what actually runs when you execute the script
if __name__ == "__main__":
    print("Starting NPC Personality System...")
    
    # Get API key and set it
    api_key = get_api_key()
    openai.api_key = api_key
    
    # Start the NPC system
    npc_personality_system()