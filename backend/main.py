import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="公司知识库RAG系统")

# ========== 全局变量 ==========
vector_db = None
rag_ready = False

# 本地Embedding（不需要阿里云密钥，直接用）
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 文本切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

UPLOAD_DIR = "uploads"
DB_DIR = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


# ========== 初始化向量库 ==========
def init_rag():
    global vector_db
    try:
        # 尝试加载已保存的向量库
        if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
            vector_db = FAISS.load_local(
                DB_DIR, embeddings, allow_dangerous_deserialization=True
            )
            print("✅ 成功加载已保存的向量库")
        else:
            # 第一次运行，创建空向量库
            vector_db = FAISS.from_texts(["初始化知识库"], embeddings)
            vector_db.save_local(DB_DIR)
            print("✅ 已创建新的向量库")
    except Exception as e:
        print(f"⚠️ 向量库初始化失败: {e}")
        vector_db = None


# 启动时执行初始化
init_rag()


# ========== 健康检查接口 ==========
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_ready": rag_ready,
        "vector_db_loaded": vector_db is not None
    }


# ========== 上传文件接口 ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db, rag_ready
    try:
        # 保存上传的文件
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # 读取文件内容
        if file.filename.endswith(".txt"):
            with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
        else:
            return {"error": "目前只支持txt文件"}

        if not full_text.strip():
            return {"error": "文件内容为空"}

        # 文本切分
        chunks = text_splitter.split_text(full_text)
        print(f"📄 切分出 {len(chunks)} 个文本块")

        # 构建/追加向量库
        if vector_db is None:
            vector_db = FAISS.from_texts(chunks, embeddings)
        else:
            vector_db.add_texts(chunks)

        # 保存到磁盘
        vector_db.save_local(DB_DIR)
        rag_ready = True

        return {
            "msg": "✅ 上传成功，知识库已更新",
            "filename": file.filename,
            "chunk_count": len(chunks)
        }

    except Exception as e:
        return {"error": f"上传失败: {str(e)}"}


# ========== 问答接口 ==========
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(req: ChatRequest):
    global vector_db
    try:
        if vector_db is None:
            return {"answer": "❌ 知识库为空，请先上传文档！"}

        # 检索相关片段
        docs = vector_db.similarity_search(req.query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # 简单拼接回答（先不用大模型，直接返回检索到的内容测试）
        answer = f"【检索到的相关内容】\n{context}\n\n【你的问题】{req.query}"

        return {
            "query": req.query,
            "answer": answer,
            "retrieved_docs": len(docs)
        }

    except Exception as e:
        return {"error": f"问答失败: {str(e)}"}


# ========== 评估接口（先放着，后面再调） ==========
class EvaluateRequest(BaseModel):
    samples: list

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    return {"msg": "评估接口已就绪，样本数: " + str(len(req.samples))}
