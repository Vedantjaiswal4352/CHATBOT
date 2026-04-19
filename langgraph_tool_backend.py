####################################################################################
                                # IMPORTS
####################################################################################
from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
import random
import requests
import os
av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
####################################################################################
                                # LLM SETUP
####################################################################################
llm_meta = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.7",   
    task="text-generation",
    max_new_tokens=1000,#256,              
    provider="auto",                 
)
model = ChatHuggingFace(llm=llm_meta)

####################################################################################
                                # TOOLS
####################################################################################
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(f_num:float,s_num:float,operation:str)-> dict:
    """
    Perform a basic arithematic operation of two numbers.
    Supported operations: add, subtract, multiple, divide.
    """
    try:
        if operation == "add":
            result = f_num + s_num
        elif operation == "subtract":
            result = f_num - s_num
        elif operation == "multiply":
            result = f_num * s_num
        elif operation == "divide":
            if s_num == 0:
                return {'error':'Division by zero is not allowed.'}
            result = f_num / s_num
        else:
            return {'error':f'Unsupported Operation "{operation}"'}
    except Exception as e:
        return {'error':str(e)}

@tool
def get_stock_price(symbol : str)->dict:
    """
    Fetch latest stock price for a given symbol (eg: 'AAPL', 'TSLA')
    Using Alpha  vantage using api key in url. 
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={av_api_key}"
    r = requests.get(url)
    return r.json()

@tool
def get_weather_data(city : str)->dict:
    """
    Fetch latest weather data for a given city (eg: 'AAPL', 'TSLA')
    Using Openweather API and its api key in url. 
    """
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}"
    r = requests.get(url)
    return r.json()

#tools list
tools = [get_stock_price,search_tool,calculator,get_weather_data]
model_with_tools = model.bind_tools(tools)

####################################################################################
                                # STATE & NODE FUNCTIONS
####################################################################################
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {'messages':[response]}

tool_node = ToolNode(tools)

####################################################################################
                                # DB CONN
####################################################################################
conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

####################################################################################
                                # NODES & EDGES
####################################################################################
graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node('tools',tool_node)
graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

####################################################################################
                                # HELPER FUNCTIONS
####################################################################################
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