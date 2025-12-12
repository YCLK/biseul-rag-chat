"""
비슬고등학교 생활규정 RAG 챗봇
- Gradio + LangChain + Google Gemini
"""

import os
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ========== 1. 데이터 로드 ==========

def load_documents():
    """CSV와 MD 파일을 Document 객체로 변환"""
    documents = []
    
    # 상점 규정 (good_rules.csv)
    df_good = pd.read_csv("good_rules.csv")
    for _, row in df_good.iterrows():
        content = f"[상점 규정] 구분: {row['구분']}, 선행내용: {row['선행내용']}, 점수: {row['점수']}점"
        documents.append(Document(page_content=content, metadata={"source": "상점규정"}))
    
    # 벌점 규정 (bad_rules.csv)
    df_bad = pd.read_csv("bad_rules.csv")
    for _, row in df_bad.iterrows():
        비고 = f", 비고: {row['비고']}" if pd.notna(row['비고']) and row['비고'] else ""
        content = f"[벌점 규정] 영역: {row['영역']}, 위반내용: {row['위반내용']}, 벌점: {row['1회 벌점']}점{비고}"
        documents.append(Document(page_content=content, metadata={"source": "벌점규정"}))
    
    # 학교 규정 (school_regulations.md)
    with open("school_regulations.md", "r", encoding="utf-8") as f:
        md_content = f.read()
    
    # MD 파일은 청크로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n"]
    )
    md_chunks = splitter.split_text(md_content)
    for chunk in md_chunks:
        documents.append(Document(page_content=chunk, metadata={"source": "학교규정"}))
    
    return documents

# ========== 2. RAG 체인 생성 ==========

def create_rag_chain(api_key: str):
    """벡터스토어와 RAG 체인 생성"""
    
    # Gemini 모델 설정
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # 문서 로드 및 벡터스토어 생성
    documents = load_documents()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # RAG 체인 생성
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    
    return chain

# ========== 3. Gradio 인터페이스 ==========

# 전역 변수
rag_chain = None

def set_api_key(api_key: str):
    """API 키 설정 및 RAG 체인 초기화"""
    global rag_chain
    if not api_key.strip():
        return "❌ API 키를 입력해주세요."
    
    try:
        rag_chain = create_rag_chain(api_key.strip())
        return "✅ API 키 설정 완료! 이제 질문할 수 있습니다."
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

def chat(message: str, history: list):
    """채팅 함수"""
    global rag_chain
    
    if rag_chain is None:
        return "⚠️ 먼저 API 키를 설정해주세요."
    
    if not message.strip():
        return "질문을 입력해주세요."
    
    try:
        # RAG 체인 실행
        result = rag_chain.invoke({"query": message})
        answer = result["result"]
        
        # 출처 정보 추가
        sources = set(doc.metadata["source"] for doc in result["source_documents"])
        source_text = ", ".join(sources)
        
        return f"{answer}\n\n📚 참고: {source_text}"
    
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# Gradio UI 구성
with gr.Blocks(title="비슬고등학교 생활규정 챗봇", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏫 비슬고등학교 생활규정 챗봇")
    gr.Markdown("학교 규칙, 상점/벌점에 대해 질문해보세요!")
    
    with gr.Row():
        api_input = gr.Textbox(
            label="Google AI API Key",
            placeholder="API 키를 입력하세요",
            type="password",
            scale=4
        )
        api_btn = gr.Button("설정", scale=1)
    
    api_status = gr.Textbox(label="상태", interactive=False)
    api_btn.click(set_api_key, inputs=api_input, outputs=api_status)
    
    chatbot = gr.ChatInterface(
        fn=chat,
        examples=[
            "욕설을 하면 벌점이 몇 점이야?",
            "상점은 어떻게 받을 수 있어?",
            "등교 시간이 몇 시야?",
            "휴대폰 사용 규칙이 뭐야?",
            "벌점 30점 받으면 어떻게 돼?"
        ],
        retry_btn=None,
        undo_btn=None,
    )

if __name__ == "__main__":
    demo.launch()
