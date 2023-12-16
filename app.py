import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Load necessary data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message(event=None):
    msg = entry_field.get().strip()
    if msg:
        entry_field.delete(0, tk.END)

        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, msg + "\n", 'user_response')
        chat_window.insert(tk.END, "\n", 'user_response')  # Add an extra line between user and bot responses
        chat_window.tag_configure('user_response', justify='right', foreground='#bf5547', font=('Arial', 12, 'bold'))

        ints = predict_class(msg)
        res = get_response(ints, intents)
        chat_window.insert(tk.END, res + "\n", 'bot_response')
        chat_window.insert(tk.END, "\n", 'bot_response')
        chat_window.tag_configure('bot_response', justify='left', foreground='#4775bf', font=('Arial', 12, 'bold'))

        chat_window.config(state=tk.DISABLED)
        chat_window.yview(tk.END)

        if msg.lower() in ["bye", "goodbye"]:
            root.after(3000, root.destroy)  # Schedule window closure after 3 seconds


root = tk.Tk()
root.title("BreastBot")

# Use a frame for the chat window to control the background color
frame = tk.Frame(root, bg='#f0f0f0')
frame.pack(expand=True, fill='both', padx=10, pady=10)

heading_label = tk.Label(frame, text="BreastBot", font=('Arial', 14, 'bold'), fg='black', bg='#f0f0f0')
heading_label.pack(pady=10)

# Create the chat window with a monospaced font for better alignment
chat_window = scrolledtext.ScrolledText(frame, width=50, height=20, wrap=tk.WORD, bg='#ffffff', font=('Courier New', 10))
chat_window.pack(expand=True, fill='both')
chat_window.config(state=tk.DISABLED, borderwidth=1, relief="solid")

# Style the user and bot messages
chat_window.tag_configure('user_response', justify='right', foreground='#0084FF', font=('Courier New', 10, 'bold'))
chat_window.tag_configure('bot_response', justify='left', foreground='#58B662', font=('Courier New', 10, 'bold'))

# Entry field for user input
entry_field = tk.Entry(frame, width=40, font=('Courier New', 10))
entry_field.pack(side=tk.LEFT, expand=True, fill='x', padx=(0, 10), pady=10)

# Send button for user to submit their message
send_button = tk.Button(frame, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT, padx=10, pady=10)
send_button.config(font=('Courier New', 10), bg='#0084FF', fg='white', relief="raised", borderwidth=2)

entry_field.bind("<Return>", send_message)

# Start the main loop
root.mainloop()
