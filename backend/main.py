import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import dashscope
# 新增解析库
import PyPDF2
from docx import Document
import json
from datetime import datetime
# 新增ollama导入
from langchain_ollama import OllamaLLM

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
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
# ===================== 【这里是我们新增的元数据部分】=====================
META_FILE = os.path.join(DB_DIR, "metadata.json")
file_meta = []
def save_meta():
    """把元数据列表写入json文件"""
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(file_meta, f, ensure_ascii=False, indent=2)
# 加载已有元数据
if os.path.exists(META_FILE):
    with open(META_FILE, "r", encoding="utf-8") as f:
        file_meta = json.load(f)
else:
    save_meta()
# ========================================================================
# ========== 【新增：解析函数】 ==========
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
        # 尝试加载已保存的向量库
        if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR))>0:
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
# ========== 【修改后的上传接口：加入元数据记录】 ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_db, rag_ready, file_meta
    try:
        # 保存上传的文件
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        
        # 根据后缀选择解析器
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
        # 文本切分
        chunks = text_splitter.split_text(full_text)
        chunk_cnt = len(chunks)
        print(f"📄 切分出 {chunk_cnt} 个文本块")
        # 记录：新增这批切片之前，向量库已有多少条
        old_size = vector_db.index.ntotal if vector_db else 0
        # 构建/追加向量库
        if vector_db is None:
            vector_db = FAISS.from_texts(chunks, embeddings)
        else:
            vector_db.add_texts(chunks)
        # 保存到磁盘
        vector_db.save_local(DB_DIR)
        rag_ready = True
        # =====【新增：写入这条文件的元信息】=====
        new_record = {
            "filename": file.filename,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_count": chunk_cnt,
            "start_id": old_size,
            "end_id": old_size + chunk_cnt - 1
        }
        file_meta.append(new_record)
        save_meta()
        # ========================================
        return {
            "msg": "✅ 上传成功，知识库已更新",
            "filename": file.filename,
            "chunk_count": chunk_cnt
        }
    except Exception as e:
        return {"error": f"上传失败: {str(e)}"}
@app.get("/list")
async def get_file_list():
    global file_meta
    return {
        "code": 200,
        "data": file_meta
    }
from pydantic import BaseModel
class DeleteItem(BaseModel):
    filename: str
@app.post("/delete")
async def delete_file(item: DeleteItem):
    global vector_db, file_meta
    target_name = item.filename
    # 找到这条记录
    find_record = None
    remain_list = []
    for one in file_meta:
        if one["filename"] == target_name and find_record is None:
            find_record = one
        else:
            remain_list.append(one)
    if find_record is None:
        return {"code":400, "msg":"找不到这个文件"}
    start = find_record["start_id"]
    end = find_record["end_id"]
    # 取出需要保留的doc（LangChain的方式，doc和index顺序一一对应）
    all_docs = vector_db.docstore._dict
    keep_docs = []
    ids_list = list(all_docs.keys())
    for i, doc_id in enumerate(ids_list):
        if not (start <= i <= end):
            keep_docs.append(all_docs[doc_id])
    # 用剩下的文档新建向量库
    if len(keep_docs) == 0:
        vector_db = FAISS.from_texts([""], embeddings)
    else:
        vector_db = FAISS.from_documents(keep_docs, embeddings)
    # 更新元数据、保存文件
    file_meta = remain_list
    with open(os.path.join("faiss_index","metadata.json"),"w",encoding="utf-8") as f:
        json.dump(file_meta, f, ensure_ascii=False, indent=2)
    vector_db.save_local("faiss_index")
    return {"code":200, "msg":f"成功删除：{target_name}"}
@app.post("/clear")
async def clear_all():
    global vector_db, file_meta
    # 新建空向量库
    vector_db = FAISS.from_texts([""], embeddings)
    file_meta = []
    with open(os.path.join("faiss_index","metadata.json"),"w",encoding="utf-8") as f:
        json.dump(file_meta, f, ensure_ascii=False, indent=2)
    vector_db.save_local("faiss_index")
    return {"code":200, "msg":"全部知识库已经清空"}
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
        # 优化Prompt，强制提炼总结
        prompt = """
你是企业知识库助手，请严格依据【参考文档】内容回答用户问题。
要求：
1. 只能使用文档内信息，禁止编造不存在的内容；
2. 精简提炼答案，不要直接复制大段原文；
3. 不要输出检索标记、序号、多余联系方式；
4. 如果文档没有相关信息，直接回复：知识库暂时没有该问题相关资料。
【参考文档】
{context}
用户问题：{query}
""".format(context=context, query=req.query)
        # ========= Ollama本地模型，替换掉原来dashscope =========
        llm = OllamaLLM(model="qwen2.5:1.5b", base_url="http://localhost:11434")
        answer = llm.invoke(prompt)
        print(f"ollama返回：{answer}")
        # =======================================================
        return {
            "query": req.query,
            "answer": answer,
            "retrieved_docs": len(docs)
        }
    except Exception as e:
        print(f"LLM调用异常：{str(e)}")
        return {"error": f"问答失败: {str(e)}"}
# ========== 评估接口（先放着，后面再调） ==========
class EvaluateRequest(BaseModel):
    samples: list
@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    return {"msg": "评估接口已就绪，样本数: " + str(len(req.samples))}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
