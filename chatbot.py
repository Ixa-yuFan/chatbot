import os
import streamlit as st
from dotenv import load_dotenv                   #本地开发读.ENV文件
from openai import OpenAI                        #引入OpenAI客户端类，用来调用DeepSeek的API

st.set_page_config(
    page_title="夏宇凡的AI助手",
    page_icon="🤖",
    layout = "centered"
)


st.markdown("""
<style>                                 
    [data-testid="stSidebar"] {
        min-width: 180px;
        max-width: 180px;
    }
</style>
""", unsafe_allow_html=True)  #style 和 /style：HTML标签，让浏览器把中间这段代码按照CSS来解析而不是文本内容

load_dotenv()
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    api_key = os.environ.get("DEEPSEEK_API_KEY")

def trim_messages(messages,max_rounds = 10):
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    maxmsgs = max_rounds * 2
    if len(other_msgs) > maxmsgs:
        other_msgs = other_msgs[-maxmsgs:]
    return system_msgs + other_msgs

client = OpenAI(                                 #创建API客户端,key + API地址
    api_key= api_key,
    base_url="https://api.deepseek.com"
)

st.title("夏宇凡的AI chatbot")                     #网站标题
with st.sidebar:
    st.header("设置")
    if st.button("清空对话"):
        st.session_state.messages = [
            {"role":"system","content":"你是一个可爱的小女孩"}
        ]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"system","content":"你是一个可爱的小女孩"}
    ]

print("AI assistant have started,input quit or exit to stop")

if len(st.session_state.messages) <= 1:
    st.markdown("##### 🤗 你好！我是你的AI助手")
    st.markdown("有什么可以帮你的嘛？我可是无所不知哦")
for msg in st.session_state.messages:               #创建聊天气泡，页面刷新也能看到聊天记录
    if msg["role"] != "system":
        avatar = "🧑" if msg["role"] != "system" else "🤖"
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if prompt := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    full_reply = ""

    st.session_state.messages = trim_messages(st.session_state.messages)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(              #调用API
                model = "deepseek-chat",
                messages = st.session_state.messages,
                stream = True
            )
        except Exception as e:
            st.error(f"出错了:{e}")
            full_reply  = "抱歉，我暂时无法回答"

        placeholder = st.empty()
        for chunk in response:
            if chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                full_reply = full_reply + piece
                placeholder.write(full_reply)

    st.session_state.messages.append({"role":"assistant","content":full_reply})

st.caption("由DEEPSEEK驱动 · 仅供学习使用")