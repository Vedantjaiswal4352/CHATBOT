from langgraph.graph import StateGraph,START, END
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

llm_openai = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",   # switch to the new model
    task="text-generation",
    max_new_tokens=1000,#256,              # increase token output
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",                 # let HF choose best provider
)
model = ChatHuggingFace(llm=llm_openai)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages':[response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)