import streamlit as st
from openai import OpenAI
import re
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from config import OPENAI_API_KEY

# Initiate API client
client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

# Title of the webpage
st.title("Health and Wellness Chatbot")

# Initialize chat history
context = "Hello! How can I help you today?"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": f"{context}"}]

# Set image under title
st.image('USC_image/USC_top_image.png', use_column_width=True)

# Handle initial interaction separately
prompt = None
response = None

# Obtains a response from the regular gpt 3.5 model
def get_completion_regular(messages, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,)
    return response.choices[0].message.content

# Obtains a response from the fine-tuned model on the FAQ dataset
def get_completion_from_messages(messages, model="ft:gpt-3.5-turbo-0125:personal::9JoHdDgX", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,)
    return response.choices[0].message.content

# Obtain the sentiment of the message from the fine-tuned model on the corpus dataset
def get_sentiment_from_messages(messages, model="ft:gpt-3.5-turbo-0125:personal::9Jtu2wwl", temperature=0):

    prompt = [{
        "role": "system",
        "content": f"""
            You are a friendly and polite health and wellness chatbot. 
            Based on the conversation, determine whether the sentiment of the user's situation is negative or positive.
            Chat history: {messages}
    """
    }]
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,)
    return response.choices[0].message.content


# Prompts the API to generate 15 websites along with their descriptions and process to JSON file format
def website_parse():

    online_search_prompt = [{
        "role": "system",
        "content": f"""
            Based on the chat history, find and provide 15 websites offering relevant advice or services to the user. 
            Store the names of the websites and a brief description of their services together in JSON format.
            The JSON format should have "name:" and "description:" elements.
            Chat history: {st.session_state.chat_history}
        """
    }]

    response_websites = get_completion_regular(online_search_prompt)
    print(response_websites)

    # Extracts the websites and descriptions into a list
    websites = re.findall(r'"name":\s*"([^"]+)"', response_websites)
    descriptions = re.findall(r'"description":\s*"([^"]+)"', response_websites)

    print("websites", websites)
    print("descriptions", descriptions)

    # Create the JSON dataset
    def create_JSON_dataset(website_name, description):
        return {
            "website": website_name,
            "description": description
        }

    # Write the dataset in a json file
    with open("online_resources.json", "w") as f:
        for i in range(len(websites)):
            example_str = json.dumps(create_JSON_dataset(websites[i], descriptions[i]))
            f.write(example_str + "\n")


# Recommends the most relevant website to the user based on the similarity between
# the user inquiry and the website descriptions
def resource_rec():
    # Function to load website descriptions from a JSON file
    def load_website_data(json_filepath):
        with open(json_filepath, 'r') as file:
            return [json.loads(line) for line in file]

    # Function to embed text using a pre-trained transformer model
    def embed_text(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            vectors = model(**inputs).pooler_output
        return vectors.numpy()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Load website data
    websites_data = load_website_data("online_resources.json")

    # Embed website descriptions
    website_embeddings = np.vstack([embed_text(site['description'], tokenizer, model) for site in websites_data])

    # User query
    user_query = "I feel stressed and need relaxation techniques."
    query_embedding = embed_text(user_query, tokenizer, model)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, website_embeddings).flatten()

    # Find the most relevant website
    max_index = similarities.argmax()
    ret_wesbite = ""
    ret_desc = ""
    if similarities[max_index] > 0.6:  # Threshold for relevance
        ret_wesbite = websites_data[max_index]['website']
        ret_desc = websites_data[max_index]['description']
        print(f"Recommended website: {websites_data[max_index]['website']}")
        print(f"Description: {websites_data[max_index]['description']}")
        print(f"Similarity score: {similarities[max_index]}")
    else:
        print("No relevant websites found. Please try a general Google search.")

    return ret_wesbite, ret_desc

# When "Enter" is pressed, the API will be prompted to give a valid response
# based on the topic, chat history, and user inquiry
def process_input():

    # Append the unser input to the chat history
    st.session_state.chat_history.append({"role": "user", "content": f"{user_input}"})

    collected_info = [{
        "role": "assistant",
        "content": f""" 
            You are a friendly and polite health and wellness chatbot assistant. 
            Continue the conversation to answer the user's inquiry.
            Provide 5 pieces of advice to the user that is relevant to most recent inquiry.
            Your response should be at most 5 sentences.
            Topic: {selected_topic}
            Chat history: {st.session_state.chat_history}
            Most recent inquiry: {user_input}
            """
    }]

    more_info_q = "If you want a recommendation to an online resource for more information, Type 'yes'."

    # If the more_info_q was asked and the user said "yes", give the user a website recommendation
    if st.session_state.chat_history[-2]["content"] == more_info_q and "yes" in user_input:
        website_parse()
        response_website, response_desc = resource_rec()

        if response_website == "":
            st.session_state.chat_history.append({"role": "assistant", "content": "No relevant websites found. Please try a general Google search."})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Visit {response_website} for more information. {response_desc}"})
    # or else give the user advice and ask if they require more information
    else:
        response_advice = get_completion_from_messages(collected_info)
        sentiment = get_sentiment_from_messages(st.session_state.chat_history).lower()
        response_advice = response_advice + "\n\n" + f"The severity of the situation is {sentiment}."

        st.session_state.chat_history.append({"role": "assistant", "content": f"{response_advice}"})
        st.session_state.chat_history.append({"role": "assistant", "content": f"{more_info_q}"})

# Define the options for the dropdown menu
options = ["General", "Medical Conditions", "Diet Plans", "Workout Routines"]

# Create a dropdown menu and get the user's selection
selected_topic = st.selectbox("Choose a topic:", options)

# Text box for user input
user_input = st.text_input("Start chatting:", key="user_input_box")

# The Enter and Clear buttons
col1, col2 = st.columns([36, 11])

# Process the input if it is not empty
with col1:
    if st.button("Enter") and user_input != "":
        process_input()

# Clears the chat history variable
with col2:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = [{"role": "assistant", "content": f"{context}"}]

# Subheader for the chat history
st.subheader("Chat history:")

# Assign the roles, user or Chatbot, to the content of the chat history
# This is only for formatting purposes
chat_history_text = ""
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        chat_history_text += "**User**: " + chat["content"] + "\n\n"
    elif chat["role"] == "assistant":
        chat_history_text += "**Chatbot**: " + chat["content"] + "\n\n"

# Creates the chat history text box and displays the chat history
st.markdown(chat_history_text, unsafe_allow_html=True)
