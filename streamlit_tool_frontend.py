import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads,delete_thread
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid
import time
################################################## Utility Function #####################################

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])




################################################## Session ##############################################

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

################################################## Sidebar ##############################################

st.sidebar.title("CHATBOT")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state['chat_threads']:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_message = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'ai'
            temp_message.append({'role':role,'content':msg.content})

        st.session_state['message_history'] = temp_message

thread_id = st.sidebar.selectbox("Select thread", retrieve_all_threads())

if st.sidebar.button("Delete this thread"):
    delete_thread(thread_id)
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    st.sidebar.success(f"Thread {thread_id} deleted.")
    time.sleep(3)
    st.rerun()


#########################################################################################################

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here...')
if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        'configurable':{'thread_id':st.session_state['thread_id']},
        'metadata':{
            'thread_id':st.session_state['thread_id']
        },
        'run_name':'chat_run'
    }
    
    with st.chat_message('ai'):
        status_holder = {"box":None}
        def ai_only_stream():
            for message_chunk,metadata in chatbot.stream(
                {"messages":[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                if isinstance(message_chunk,ToolMessage):
                    tool_name = getattr(message_chunk,"name","tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state = "running",
                            expanded=True
                        )
                if isinstance(message_chunk,AIMessage):
                    yield message_chunk.content
        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label = "✅ Tool finished", state="complete", expanded=False
            )



    st.session_state['message_history'].append({'role':'ai','content':ai_message})
