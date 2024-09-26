from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

import warnings 
warnings.filterwarnings("ignore")

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate response using Hugging Face GPT-2 model
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, do_sample=True, top_p=0.95, top_k=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function for sentiment analysis
def analyze_sentiment(response_text):
    analysis = TextBlob(response_text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return {"response": response_text, "polarity": polarity, "subjectivity": subjectivity, "sentiment": sentiment}

# Function to interact with user
def user_interaction():
    user_input = input("You: ")
    gpt_response = generate_response(user_input)
    print(f"GPT-2: {gpt_response}")
    sentiment = analyze_sentiment(gpt_response)
    return sentiment

# Collect multiple interactions and analyze sentiment
def collect_conversations(num_interactions):
    conversations = []
    for i in range(num_interactions):
        print(f"Interaction {i + 1}")
        sentiment = user_interaction()
        conversations.append(sentiment)
    return pd.DataFrame(conversations)

# Generate sentiment report with visualizations
def generate_report(df):
    sns.set(style="whitegrid")
    
    # Plotting polarity
    plt.figure(figsize=(10, 6))
    sns.histplot(df['polarity'], bins=10, kde=True)
    plt.title('Distribution of Sentiment Polarity')
    plt.xlabel('Polarity')
    plt.ylabel('Frequency')
    plt.show()

    # Plotting subjectivity
    plt.figure(figsize=(10, 6))
    sns.histplot(df['subjectivity'], bins=10, kde=True, color='orange')
    plt.title('Distribution of Subjectivity')
    plt.xlabel('Subjectivity')
    plt.ylabel('Frequency')
    plt.show()

    # Sentiment pie chart
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral', 'lightblue'])
    plt.title('Sentiment Distribution')
    plt.show()

# Main program execution
if __name__ == "__main__":
    num_interactions = int(input("How many interactions would you like? "))
    conversation_df = collect_conversations(num_interactions)
    print("\nConversation Data:\n", conversation_df)
    
    # Generate sentiment report
    generate_report(conversation_df)
