import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent that gives precise and to-the-point answers."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

model= ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),      #nvidia/nemotron-nano-9b-v2   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

chain = prompt | model

history_dir = Path("chat_sessions")
history_dir.mkdir(exist_ok=True)

def get_session_history(session_id: str):
    return FileChatMessageHistory(
        file_path=str(history_dir / f"{session_id}.json")
    )

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="chat_history",
)

print("Chatbot started! Type 'exit' to quit.\n")

session_id = input("Enter session ID (or press Enter for 'default'): ").strip() or "default"
config = {"configurable": {"session_id": session_id}}

print(f"\nUsing session: {session_id}")
print("=" * 50)

while True:
    user_input = input("\nYOU: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    if not user_input.strip():
        continue
    
    
    response = chain_with_history.invoke(
        {"query": user_input},
        config=config
    )
    
    print(f"AI: {response.content}")