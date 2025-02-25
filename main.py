from typing import Dict
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import re


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


def fetch_hf_response(prompt: str, model_name: str = "gpt2") -> str:
    """
    Use Hugging Face GPT-2 to generate a response for the given prompt.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(inputs, max_length=512, num_return_sequences=1, temperature=0.7)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_personality_profile(world_description: str, npc_role: str) -> str:
    """
    Generate the prompt for Hugging Face's GPT-2 to create NPC personality.
    """
    prompt = f"""
    You are tasked with creating an NPC in a fictional world. 
    World Description: {world_description}.
    Role: {npc_role}.
    Provide personality traits (Openness, Conscientiousness, Agreeableness, Neuroticism, Extraversion) 
    on a scale from 0 to 100, and a brief description of the NPC's behavior.
    """
    return fetch_hf_response(prompt, model_name="gpt2")


def parse_personality_from_response(response: str) -> Personality:
    """
    Extracts personality traits from the Hugging Face GPT-2 response string.
    """
    try:
        openness = int(re.search(r"Openness: (\d+)", response).group(1))
        conscientiousness = int(re.search(r"Conscientiousness: (\d+)", response).group(1))
        agreeableness = int(re.search(r"Agreeableness: (\d+)", response).group(1))
        neuroticism = int(re.search(r"Neuroticism: (\d+)", response).group(1))
        extraversion = int(re.search(r"Extraversion: (\d+)", response).group(1))
        return Personality(
            openness=openness,
            conscientiousness=conscientiousness,
            agreeableness=agreeableness,
            neuroticism=neuroticism,
            extraversion=extraversion,
        )
    except (AttributeError, ValueError) as e:
        raise ValueError(f"Failed to parse personality from response: {response}. Error: {e}")


RESPONSES_BEHAVIORS = {
    "low_extraversion": "The NPC hesitates and gives you a short response.",
    "high_extraversion": "The NPC enthusiastically engages in a long conversation with you.",
}


def generate_dynamic_response(npc_personality: Personality) -> str:
    """
    Generate a predefined response based on NPC's extraversion level.
    """
    if npc_personality.extraversion < 30:
        return RESPONSES_BEHAVIORS["low_extraversion"]
    else:
        return RESPONSES_BEHAVIORS["high_extraversion"]


def dynamic_interaction(
        npc_personality: Personality, scene_description: str, conversation_history: str, model_name: str = "gpt2"
) -> str:
    """
    Use Hugging Face to dynamically interact with the NPC based on personality, scene, and history.
    """
    prompt = (
        f"Scene: {scene_description}\n"
        f"NPC Personality: {npc_personality}\n"
        f"Conversation History: {conversation_history}\n"
        f"Continue the conversation:"
    )
    try:
        return fetch_hf_response(prompt, model_name=model_name)
    except Exception as e:
        return f"Error: {str(e)}"


def npc_personality_system() -> None:
    world_description = "A medieval fantasy world with kingdoms and magic."
    npc_role = "Village healer"

    npc_profile_description = generate_personality_profile(world_description, npc_role)
    try:
        npc_personality = parse_personality_from_response(npc_profile_description)
    except ValueError as e:
        print(f"Error creating NPC personality: {e}")
        return

    print("Entering a conversation with the NPC. Type 'exit' to leave.")
    conversation_history = ""
    while True:
        player_input = input("You: ")
        if player_input.lower() == 'exit':
            print("You left the conversation.")
            break
        response = dynamic_interaction(npc_personality, world_description, conversation_history, model_name="gpt2")
        print(f"NPC: {response}")
        conversation_history += f"\nPlayer: {player_input}\nNPC: {response}"


# Example NPCs with predefined personalities
npc_1 = Personality(openness=70, conscientiousness=50, agreeableness=60, neuroticism=40, extraversion=80)
npc_2 = Personality(openness=30, conscientiousness=70, agreeableness=40, neuroticism=70, extraversion=20)
game_world: Dict[str, Personality] = {
    "blacksmith": npc_1,
    "merchant": npc_2,
}
