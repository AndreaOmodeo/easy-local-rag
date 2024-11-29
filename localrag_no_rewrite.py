import torch
import ollama
import os
from openai import OpenAI
import argparse
import pickle

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def save_embeddings_to_file(embeddings, filename):
    """Saves embeddings to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings_from_file(filename):
    """Loads embeddings from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
            return embeddings
    except FileNotFoundError:
        print(f"Error: Embeddings file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

# Function to interact with the Ollama model
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    # Get relevant context from the vault
    relevant_context = get_relevant_context(user_input, vault_embeddings_tensor, vault_content, top_k=6)
    if relevant_context:
        # Convert list to a single string with newlines between items
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    # Prepare the user's input by concatenating it with the relevant context
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = context_str + "\n\n" + user_input
    
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input_with_context})
    
    # Create a message history including the system message and the conversation history
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    # Send the completion request to the Ollama model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    # Return the content of the response from the model
    return response.choices[0].message.content

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3.1", help="Ollama model to use (default: llama3.1)")
args = parser.parse_args()

# Configuration for the Ollama API client
client = OpenAI(
    #base_url='http://192.168.1.60:11434/v1',
    base_url='http://pclouderae02:8080/v1',
    api_key='llama3.2'
)

vault_path = "vault.txt"  # Or whatever your vault file path is
embeddings_path = "vault.pickle" # Or whatever your embeddings file path is
# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists(vault_path):
    with open(vault_path, "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
if os.path.exists(embeddings_path) and os.path.getmtime(embeddings_path) > os.path.getmtime(vault_path):
    print(f"Loading embeddings from '{embeddings_path}'...")
    vault_embeddings = load_embeddings_from_file(embeddings_path)
else:
    print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
    vault_embeddings = []
    for content in vault_content:
        print('.', end='', flush=True)
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    save_embeddings_to_file(vault_embeddings, embeddings_path)

# Convert to tensor and print embeddings
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# Conversation loop
conversation_history = []
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text"

while True:
    user_input = input(YELLOW + "Ask a question about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break

    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
