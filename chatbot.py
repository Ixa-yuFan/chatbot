import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def trim_messages(messages,max_rounds = 10):
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    maxmsgs = max_rounds * 2
    if len(other_msgs) > maxmsgs:
        other_msgs = other_msgs[-maxmsgs:]
    return system_msgs + other_msgs

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

st.title("Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"system","content":"编程导师"}
    ]

print("AI assistant have started,input quit or exit to stop")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if prompt := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    full_reply = ""

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model = "deepseek-flash",
            messages = st.session_state.messages,
            stream = True
        )

        placeholder = st.empty()
        for chunk in response:
            if chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                full_reply = full_reply + piece
                placeholder.write(full_reply)

    st.session_state.messages.append({"role":"assistant","content":full_reply})
