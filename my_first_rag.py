# ===================== 仅需修改这里 =====================
YOUR_QWEN_API_KEY = "你的API"
DOC_PATH = "./data"
VECTOR_INDEX_PATH = "./my_rag_index"
# ========================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import dashscope

# ===================== 嵌入模型（修复版，不报错） =====================
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ 嵌入模型初始化完成")

# ===================== 向量库加载 =====================
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

if os.path.exists(VECTOR_INDEX_PATH) and os.listdir(VECTOR_INDEX_PATH):
    print("🔍 加载本地向量库...")
    vector_db = FAISS.load_local(
        VECTOR_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("✅ 本地向量库加载成功")
else:
    print("📄 首次构建向量库...")
    loader = DirectoryLoader(
        path=DOC_PATH,
        glob=["**/*.md", "**/*.txt", "**/*.pdf", "**/*.docx", "**/*.xlsx"],
        show_progress=True,
        use_multithreading=True,
        silent_errors=True
    )
    documents = loader.load()
    print(f"✅ 文档加载完成：{len(documents)} 个")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)

    vector_db = FAISS.from_documents(split_docs, embedding_model)
    vector_db.save_local(VECTOR_INDEX_PATH)
    print("✅ 向量库构建并保存成功")

# ===================== RAG 问答 =====================
dashscope.api_key = YOUR_QWEN_API_KEY

def rag_qa(user_question, top_k=3):
    relevant_docs = vector_db.similarity_search(user_question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    请你仅基于以下提供的上下文内容，回答用户的问题。
    如果上下文里没有相关内容，直接回答「我没有找到相关的信息，无法回答这个问题」，绝对不要编造内容。
    
    上下文内容：
    {context}
    
    用户问题：{user_question}
    """
    
    response = dashscope.Generation.call(
        model=dashscope.Generation.Models.qwen_turbo,
        prompt=prompt,
        temperature=0.3
    )
    return response.output.text, relevant_docs

# ===================== 测试 =====================
if __name__ == "__main__":
    print("\n🎉 工业级 RAG 系统运行成功！")
    print("="*60)
    test_question = "番茄炒蛋怎么做"
    answer, source = rag_qa(test_question)
    print(f"问题：{test_question}")
    print(f"回答：{answer}")
        # 测试2：跨文档检索
    test_question2 = "可乐鸡翅怎么做"
    answer2, source2 = rag_qa(test_question2)
    print(f"问题2：{test_question2}")
    print(f"回答：{answer2}")