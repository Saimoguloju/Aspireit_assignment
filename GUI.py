import tkinter as tk
from tkinter import scrolledtext

def on_generate_response():
    user_input = user_entry.get("1.0", "end-1c")  # Get input from the user
    gpt_response = generate_response(user_input)  # Generate response
    sentiment = analyze_sentiment(gpt_response)  # Perform sentiment analysis
    
    # Show the generated response and sentiment
    response_display.insert(tk.END, f"You: {user_input}\nGPT-2: {gpt_response}\nSentiment: {sentiment['sentiment']}\n\n")

# Create a GUI window
window = tk.Tk()
window.title("GPT-2 Chat with Sentiment Analysis")

# User input area
user_entry = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=10)
user_entry.grid(row=0, column=0, padx=10, pady=10)

# Response display area
response_display = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=15)
response_display.grid(row=1, column=0, padx=10, pady=10)

# Generate button
generate_button = tk.Button(window, text="Generate Response", command=on_generate_response)
generate_button.grid(row=2, column=0, padx=10, pady=10)

# Run the GUI loop
window.mainloop()
