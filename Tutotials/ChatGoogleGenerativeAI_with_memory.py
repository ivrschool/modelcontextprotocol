#!/usr/bin/env python

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # Wrapper for the Google Gemini API via LangChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage  # For defining messages
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import openai
# Load environment variables (make sure GOOGLE_API_KEY is in your .env)
load_dotenv()

# Initialize the Google Gemini LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_retries=2,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )
#
os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_retries=2,
)

memory = ConversationBufferMemory()
conversation =  ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Start chat loop
print("Chat started. Type 'exit' to quit.\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = conversation.predict(input=user_input)

    print(f"Assistant: {response}\n")






# # Initialize chat history as a list of messages
# chat_history = []

# # Optional: You can add a system message if you want to guide behavior
# # system_message = SystemMessage(content="You are a helpful assistant.")
# # chat_history.append(system_message)

# # Start chat loop
# print("Chat started. Type 'exit' to quit.\n")

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         break

#     # Add user message to history
#     chat_history.append(HumanMessage(content=user_input))

#     # Call LLM with full chat history
#     response = llm.invoke(chat_history)

#     # Add AI response to history
#     chat_history.append(AIMessage(content=response.content))

#     # Print AI response
#     print(f"Assistant: {response.content}\n")
