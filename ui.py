import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag_pdf import ChatPDF
from rag_web import ChatWEB

st.set_page_config(page_title="ChatPDF and ChatWeb")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def read_and_process_link():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    link = st.session_state["link_input"]

    with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {link}"):
        st.session_state["assistant"].ingest(link)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = None

    st.header("ChatPDF and ChatWeb")

    input_type = st.radio("Select input type", options=["PDF", "Web Link"], index=0)

    if input_type == "PDF":
        st.subheader("Upload a document")
        st.file_uploader(
            "Upload document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        if not isinstance(st.session_state.get("assistant"), ChatPDF):
            st.session_state["assistant"] = ChatPDF()

    elif input_type == "Web Link":
        st.subheader("Enter a web link")
        st.text_input("Web link", key="link_input")
        st.button("Process Link", on_click=read_and_process_link)
        if not isinstance(st.session_state.get("assistant"), ChatWEB):
            st.session_state["assistant"] = ChatWEB()

    st.session_state["ingestion_spinner"] = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
