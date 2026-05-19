####################################################################################
                                # IMPORTS
####################################################################################
from  __future__ import annotations
from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Annotated,Any, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import os
import tempfile
av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
####################################################################################
# LAZY LOADING SINGLETONS
####################################################################################

_LLM = None
_MODEL = None
_EMBEDDINGS = None


def get_llm():
    global _LLM

    if _LLM is None:
        print("Loading HuggingFace endpoint...")

        _LLM = HuggingFaceEndpoint(
            repo_id="MiniMaxAI/MiniMax-M2.7",
            task="text-generation",
            max_new_tokens=1000,
            provider="auto",
        )

    return _LLM


def get_chat_model():
    global _MODEL

    if _MODEL is None:
        print("Loading chat model...")

        _MODEL = ChatHuggingFace(
            llm=get_llm()
        )

    return _MODEL


def get_embeddings():
    global _EMBEDDINGS

    if _EMBEDDINGS is None:
        print("Loading embeddings model...")

        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _EMBEDDINGS

####################################################################################
                                # PDF RETRIEVER
####################################################################################
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Any] = {}

# def _get_retriever(thread_id: Optional[str]):
#     """Fetch the retriever for the thread if available.""" 
#     if thread_id and thread_id in _THREAD_RETRIEVERS:
#         return _THREAD_RETRIEVERS[thread_id]
#     return None

def ingest_pdf(file_bytes, thread_id: str, filename:Optional[str] = None)-> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Return a summary dict that can be surfaced in th UI. 
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    FAISS_BASE_PATH = "/home/site/wwwroot/faiss_indexes"

    # Create FAISS storage folder
    os.makedirs(FAISS_BASE_PATH, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # vector_store = FAISS.from_documents(chunks, get_embeddings())
        faiss_path = f"{FAISS_BASE_PATH}/{thread_id}"

        if os.path.exists(faiss_path):
            vector_store = FAISS.load_local(
                faiss_path,
                get_embeddings(),
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)

        else:
            vector_store = FAISS.from_documents(
                chunks,
                get_embeddings()
            )

        vector_store.save_local(faiss_path)
        # Save updated vector store
        vector_store.save_local(faiss_path)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k":4}
        )
        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[thread_id]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

def _get_retriever(thread_id: Optional[str]):
    FAISS_BASE_PATH = "/home/site/wwwroot/faiss_indexes"

    if not thread_id:
        return None

    # Return cached retriever
    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    faiss_path = f"{FAISS_BASE_PATH}/{thread_id}"

    # Load from disk if exists
    if os.path.exists(faiss_path):

        vector_store = FAISS.load_local(
            faiss_path,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[thread_id] = retriever

        return retriever

    return None

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
        return {
            "first_num": f_num,
            "second_num": s_num,
            "operation": operation,
            "result": result,
        }
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

@tool
def rag_tool(query:str, thread_id: Optional[str] = None)-> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool. 
    """
    thread_id = str(thread_id)
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error":"No document indexed for this chat. Upload the PDF first.",
            "query":query,
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query":query,
        "context":context,
        "metadata":metadata,
        "source_file":_THREAD_METADATA.get(str(thread_id),{}).get("filename")
    }

#tools list
tools = [get_stock_price,search_tool,calculator,get_weather_data,rag_tool]
_MODEL_WITH_TOOLS = None


def get_model_with_tools():
    global _MODEL_WITH_TOOLS

    if _MODEL_WITH_TOOLS is None:
        print("Binding tools to model...")

        _MODEL_WITH_TOOLS = get_chat_model().bind_tools(tools)

    return _MODEL_WITH_TOOLS

####################################################################################
                                # STATE & NODE FUNCTIONS
####################################################################################
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState,config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config,dict):
        thread_id = config.get("configurable",{}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message,*state['messages']]
    response = get_model_with_tools().invoke(messages,config=config)
    # inject thread_id into tool calls if missing
    if hasattr(response, "tool_calls"):
        for call in response.tool_calls:
            if call["name"] == "rag_tool" and not call["args"].get("thread_id"):
                call["args"]["thread_id"] = thread_id
    return {'messages':[response]}

tool_node = ToolNode(tools)

####################################################################################
                                # DB CONN & CHECKPOINTER
####################################################################################
DB_PATH = "/home/site/wwwroot/chatbot.db"
conn = sqlite3.connect(database=DB_PATH,check_same_thread=False)
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
# graph.add_edge('chat_node',END)

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


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})



