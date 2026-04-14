from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv
load_dotenv()

llm_meta = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",   
    task="text-generation",
    max_new_tokens=1000,#256,              
    provider="auto",                 
)
model = ChatHuggingFace(llm=llm_meta)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages':[response]}

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def delete_thread(thread_id, db_path="chatbot.db"):
    cursor = conn.cursor()
    
    try:
        # Delete messages linked to the thread
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        
        # Delete the thread itself (if you have a threads table)
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        
        conn.commit()
        print(f"Thread {thread_id} and its data deleted successfully.")
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")




# test
# response = chatbot.invoke(
#     {"messages":[HumanMessage(content="How are you?")]},
#     config={'configurable':{'thread_id':'thread-1'}}
# )

# print(response)