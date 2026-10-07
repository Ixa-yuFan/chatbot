import os
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv                   #本地开发读.ENV文件
from openai import OpenAI                        #引入OpenAI客户端类，用来调用DeepSeek的API

HISTORY_FILE = "chat_history.json"

def save_history(messages):                       #将历史对话保存到本地文件中
    with open(HISTORY_FILE, "w",encoding="utf-8") as f:
        json.dump(messages, f,ensure_ascii=False,indent=2)

def load_history():                               #将历史对话从本地文件中加载出来
    try:
        with open(HISTORY_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError,json.JSONDecodeError):
        # 第一个报错：未找到文件，
        # 第二个报错：文件为空（第二个报错在运行一次之后其实没有作用了）
        return [{"role": "system", "content":"你是一个可爱的小女孩"}]


def trim_messages(messages,max_rounds = 10):            #对话截断函数，只记录最近的十轮对话
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    maxmsgs = max_rounds * 2
    if len(other_msgs) > maxmsgs:
        other_msgs = other_msgs[-maxmsgs:]
    return system_msgs + other_msgs


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

client = OpenAI(                                 #创建API客户端,key + API地址
    api_key= api_key,
    base_url="https://api.deepseek.com"
)



# 用 DeepSeek 的 API Key 来调用 embedding 和对话模型
api_key = os.environ.get("DEEPSEEK_API_KEY")

# 1. 加载文档
def load_documents(docs_dir="docs"):
    documents = []
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())
    return documents

# 2. 切分文档
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

# 3. 存入向量库
def build_vectorstore(chunks):
    for i,chunk in enumerate(chunks[:3]):
        print(f"第{i+1}个chunk的元数据:{chunk.metadata}")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists("./chroma_db"):
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. 检索并回答
def answer_with_rag(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "只根据上下文回答，不要编造，用简洁的语言回答，不超过100字"},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"}
        ]
    )
    return response.choices[0].message.content, [doc.page_content for doc in docs]



@st.cache_resource
def get_vectorstore():
    documents = load_documents()
    chunks = split_documents(documents)
    return build_vectorstore(chunks)

st.title("夏宇凡的AI chatbot")         #网站标题

with st.sidebar:                     #网站侧边设置功能
    st.header("设置")
    mode = st.radio("模式",["普通聊天","文档问答"])
    uploaded_file = st.file_uploader("上传文档",type = ["pdf","txt"])
    if uploaded_file is not None:
        save_path = os.path.join("docs",uploaded_file.name)
        print(f"准备保存到:{save_path}")
        with open(save_path,"wb") as f:
            f.wirte(uploaded_file.getbuffer())
        st.success(f"已上传:{uploaded_file.name}")
        st.cache_resource.clear()

    if st.button("清空对话"):
        st.session_state.messages = [
            {"role":"system","content":"你是一个可爱的小女孩"}
        ]
        save_history(st.session_state.messages)
        st.rerun()                    #网站刷新

if "messages" not in st.session_state:
    st.session_state["messages"] = load_history()

if len(st.session_state.messages) <= 1:
    st.markdown("##### 🤗 你好！我是你的AI助手")
    st.markdown("有什么可以帮你的嘛？我可是无所不知哦")
for msg in st.session_state.messages:               #创建聊天气泡，页面刷新也能看到聊天记录
    if msg["role"] != "system":
        avatar = "🧑" if msg["role"] != "system" else "🤖"
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

if prompt := st.chat_input("输入你的问题"):           #给prompt赋值，值为用户输入的句子，如果用户未输入，则给prompt赋值none
    st.session_state.messages.append({"role":"user","content":prompt})
    save_history(st.session_state.messages)
    with st.chat_message("user"):
        st.write(prompt)

    full_reply = ""

    st.session_state.messages = trim_messages(st.session_state.messages)

    with st.chat_message("assistant"):
        if mode == "文档回答":
            vectorstore = get_vectorstore()
            answer,sources = answer_with_rag(prompt, vectorstore)
            st.write(answer)
            full_reply = answer
        else:
            response = client.chat.completions.create(
                model = "deepseek-chat",
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

st.caption("由DEEPSEEK驱动 · 仅供学习使用")