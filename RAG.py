import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI


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
