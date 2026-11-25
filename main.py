import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import PyPDF2
from docx import Document
from langchain_ollama import OllamaLLM

load_dotenv()
app = FastAPI(title="公司知识库RAG系统")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 全局变量 ==========
vector_db = None
rag_ready = False
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
UPLOAD_DIR = "uploads"
DB_DIR = "faiss_index"
SESSIONS_FILE = "sessions.json"
HISTORY_FILE = "chat_history.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# ========== 会话 / 历史读写（带损坏兜底） ==========
def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_sessions():
    return load_json(SESSIONS_FILE, [])

def save_sessions(data):
    save_json(SESSIONS_FILE, data)

def load_histories():
    data = load_json(HISTORY_FILE, {})
    if not isinstance(data, dict):
        return {}
    return data

def save_histories(data):
    save_json(HISTORY_FILE, data)

# ===================== 文件元数据 =====================
META_FILE = os.path.join(DB_DIR, "metadata.json")
file_meta = load_json(META_FILE, [])

def save_meta():
    save_json(META_FILE, file_meta)

# ========== 解析函数 ==========
def parse_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def parse_docx(file_path: str) -> str:
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ========== 初始化向量库 ==========
def init_rag():
    global vector_db
    try:
        if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
            vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
            print("✅ 成功加载已保存的向量库")
        else:
            vector_db = FAISS.from_texts(["初始化知识库"], embeddings)
            vector_db.save_local(DB_DIR)
            print("✅ 已创建新的向量库")
    except Exception as e:
        print(f"⚠️ 向量库初始化失败: {e}")
        vector_db = None

init_rag()

# ========== 会话接口（第六阶段新增） ==========
@app.post("/session/new")
async def new_session():
    sessions = load_sessions()
    sid = uuid.uuid4().hex[:8]
    sessions.append({
        "session_id": sid,
        "title": "新对话",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_sessions(sessions)
    histories = load_histories()
    histories[sid] = []
    save_histories(histories)
    return {"session_id": sid, "title": "新对话"}

@app.get("/session/list")
async def session_list():
    return {"sessions": load_sessions()}

class DeleteSession(BaseModel):
    session_id: str

@app.post("/session/delete")
async def delete_session(item: DeleteSession):
    sessions = load_sessions()
    sessions = [s for s in sessions if s["session_id"] != item.session_id]
    save_sessions(sessions)
    histories = load_histories()
    histories.pop(item.session_id, None)
    save_histories(histories)
    return {"msg": "✅ 会话已删除"}

@app.get("/session/history")
async def session_history(session_id: str):
    histories = load_histories()
    return {"history": histories.get(session_id, [])}

@app.post("/clear_history")
async def clear_history(session_id: str):
    histories = load_histories()
    histories[session_id] = []
    save_histories(histories)
    return {"msg": "✅ 当前会话历史已清空"}

# ========== 上传接口 ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db, rag_ready, file_meta
    try:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        filename_low = file.filename.lower()
        if filename_low.endswith(".txt"):
            full_text = parse_txt(save_path)
        elif filename_low.endswith(".pdf"):
            full_text = parse_pdf(save_path)
        elif filename_low.endswith(".docx"):
            full_text = parse_docx(save_path)
        else:
            return {"error": "目前只支持txt、pdf、docx文件"}
        if not full_text.strip():
            return {"error": "文件内容为空或者解析不出文字"}

        chunks = text_splitter.split_text(full_text)
        chunk_cnt = len(chunks)
        old_size = vector_db.index.ntotal if vector_db else 0

        if vector_db is None:
            vector_db = FAISS.from_texts(chunks, embeddings)
        else:
            vector_db.add_texts(chunks)
        vector_db.save_local(DB_DIR)
        rag_ready = True

        file_meta.append({
            "filename": file.filename,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_count": chunk_cnt,
            "start_id": old_size,
            "end_id": old_size + chunk_cnt - 1
        })
        save_meta()

        return {"msg": "✅ 上传成功，知识库已更新", "filename": file.filename, "chunk_count": chunk_cnt}
    except Exception as e:
        return {"error": f"上传失败: {str(e)}"}

@app.get("/list")
async def get_file_list():
    return {"code": 200, "data": file_meta}

class DeleteItem(BaseModel):
    filename: str

@app.post("/delete")
async def delete_file(item: DeleteItem):
    global vector_db, file_meta
    target_name = item.filename
    find_record = None
    remain_list = []
    for one in file_meta:
        if one["filename"] == target_name and find_record is None:
            find_record = one
        else:
            remain_list.append(one)
    if find_record is None:
        return {"code": 400, "msg": "找不到这个文件"}
    start = find_record["start_id"]
    end = find_record["end_id"]
    all_docs = vector_db.docstore._dict
    keep_docs = []
    ids_list = list(all_docs.keys())
    for i, doc_id in enumerate(ids_list):
        if not (start <= i <= end):
            keep_docs.append(all_docs[doc_id])
    if len(keep_docs) == 0:
        vector_db = FAISS.from_texts([""], embeddings)
    else:
        vector_db = FAISS.from_documents(keep_docs, embeddings)
    file_meta = remain_list
    save_meta()
    vector_db.save_local(DB_DIR)
    return {"code": 200, "msg": f"成功删除：{target_name}"}

@app.post("/clear")
async def clear_all():
    global vector_db, file_meta
    vector_db = FAISS.from_texts([""], embeddings)
    file_meta = []
    save_meta()
    vector_db.save_local(DB_DIR)
    return {"code": 200, "msg": "全部知识库已经清空"}

# ========== 问答接口（按 session_id 隔离） ==========
class ChatRequest(BaseModel):
    query: str
    session_id: str

@app.post("/chat")
async def chat(req: ChatRequest):
    global vector_db
    try:
        if vector_db is None:
            return {"answer": "❌ 知识库为空，请先上传文档！"}

        histories = load_histories()
        history = histories.get(req.session_id, [])

        # 指代问句扩写，提升检索命中率
        history_lines = [
            f"{'用户' if m.get('role') == 'user' else '助手'}：{m.get('content', '')}"
            for m in history[-10:]
        ]
        history_text_block = "\n".join(history_lines)

        rewrite_prompt = f"""
根据下面历史对话，把用户当前问句改写为完整、无代词的查询语句，只输出改写后的问题，不要别的：
【历史对话】
{history_text_block}
用户现在的问题：{req.query}
"""
        rewrite_llm = OllamaLLM(model="qwen2.5:1.5b", base_url="http://localhost:11434")
        search_query = rewrite_llm.invoke(rewrite_prompt).strip()
        print("改写后的检索query：", search_query)

        docs = vector_db.similarity_search(search_query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        history_text = ""
        if history_lines:
            history_text = "【历史对话】\n" + history_text_block + "\n"

        prompt = """
你是企业知识库助手，强制遵守下面规则，不能违反：
1. 优先使用【参考文档】里面提供的内容回答；
2. 如果参考文档为空，但【历史对话】里面已经有答案，可以直接用历史对话里面的信息回答；
3. 如果文档和历史都没有相关信息，只能原样输出这句话：知识库暂时没有该问题相关资料，除此之外不许说别的任何话；
4. 找到相关内容时，精简提炼答案，禁止直接复制大段原文；
5. 禁止编造文档不存在的信息；
6. 可以参考【历史对话】理解上下文指代（如“它”“这个”）。
{history}【参考文档】
{context}
用户问题：{query}
""".format(history=history_text, context=context, query=req.query)

        llm = OllamaLLM(model="qwen2.5:1.5b", base_url="http://localhost:11434")
        answer = llm.invoke(prompt)
        print(f"ollama返回：{answer}")

        # 写入当前会话历史
        history.append({"role": "user", "content": req.query})
        history.append({"role": "assistant", "content": answer})
        histories[req.session_id] = history
        save_histories(histories)

        # 首条问题作为会话标题
        sessions = load_sessions()
        for s in sessions:
            if s["session_id"] == req.session_id and s["title"] == "新对话":
                s["title"] = req.query[:10]
        save_sessions(sessions)

        return {"query": req.query, "answer": answer, "retrieved_docs": len(docs)}
    except Exception as e:
        print(f"LLM调用异常：{str(e)}")
        return {"error": f"问答失败: {str(e)}"}

# ========== 评估接口 ==========
class EvaluateRequest(BaseModel):
    samples: list

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    return {"msg": "评估接口已就绪，样本数: " + str(len(req.samples))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
