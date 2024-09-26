import openai
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set up OpenAI API key
openai.api_key = 'your-openai-api-key'   

# Function to generate GPT-3/4 response using the latest OpenAI API
def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" if you have access to it
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # System message instructing the model
            {"role": "user", "content": prompt}  # The user's prompt
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]['message']['content'].strip()

# Function for sentiment analysis using TextBlob
def analyze_sentiment(response_text):
    analysis = TextBlob(response_text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    return {"response": response_text, "polarity": polarity, "subjectivity": subjectivity, "sentiment": sentiment}

# Function to interact with user and generate responses
def user_interaction():
    user_input = input("You: ")
    gpt_response = generate_response(user_input)
    print(f"GPT-3/4: {gpt_response}")
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
