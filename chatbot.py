import os
import time
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key= st.secrets["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key= api_key,
    base_url="https://api.deepseek.com"
)

st.title("夏宇凡的AI chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"system","content":"热情的小女孩"}
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
