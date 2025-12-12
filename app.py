import os
import gradio as gr
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")

# 2. 문서 로드
loaders = [
    TextLoader("school_regulations.txt", encoding="utf-8"),
    #CSVLoader("bad_rules.csv", encoding="cp949"),
    #CSVLoader("good_rules.csv", encoding="cp949"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# 3. 문서 분할 및 벡터 저장소
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
retriever = vectorstore.as_retriever()

# 4. RAG 체인 생성 (구버전 방식: RetrievalQA)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# RetrievalQA는 구버전에서 가장 안정적인 체인입니다.
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False  # 답변만 받기
)

# 5. Gradio 인터페이스 함수
def predict(message, history):
    # 구버전에서는 .invoke 대신 .run을 사용하기도 했으나, 
    # 최근 버전 호환을 위해 invoke를 쓰되, 안되면 .run(message)로 바꾸세요.
    try:
        response = rag_chain.invoke(message)
        return response['result'] # RetrievalQA의 결과 키는 보통 'result' 입니다.
    except:
        return rag_chain.run(message) # 아주 구버전일 경우 대비

# 6. 앱 실행
if __name__ == "__main__":
    gr.ChatInterface(
        fn=predict,
        title="🏫 학교 생활규정 안내 챗봇 (Legacy)",
        description="학교 규칙, 상점, 벌점에 대해 물어보세요.",
        examples=["욕설을 하면 벌점이 몇 점이야?", "청소를 잘하면 상점을 받을 수 있어?"],
    ).launch()import os
import gradio as gr
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")

# 2. 문서 로드
loaders = [
    TextLoader("school_regulations.txt", encoding="utf-8"),
    #CSVLoader("bad_rules.csv", encoding="cp949"),
    #CSVLoader("good_rules.csv", encoding="cp949"),
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

# 3. 문서 분할 및 벡터 저장소
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=OPENAI_API_KEY))
retriever = vectorstore.as_retriever()

# 4. RAG 체인 생성 (구버전 방식: RetrievalQA)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# RetrievalQA는 구버전에서 가장 안정적인 체인입니다.
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False  # 답변만 받기
)

# 5. Gradio 인터페이스 함수
def predict(message, history):
    # 구버전에서는 .invoke 대신 .run을 사용하기도 했으나, 
    # 최근 버전 호환을 위해 invoke를 쓰되, 안되면 .run(message)로 바꾸세요.
    try:
        response = rag_chain.invoke(message)
        return response['result'] # RetrievalQA의 결과 키는 보통 'result' 입니다.
    except:
        return rag_chain.run(message) # 아주 구버전일 경우 대비

# 6. 앱 실행
if __name__ == "__main__":
    gr.ChatInterface(
        fn=predict,
        title="🏫 학교 생활규정 안내 챗봇 (Legacy)",
        description="학교 규칙, 상점, 벌점에 대해 물어보세요.",
        examples=["욕설을 하면 벌점이 몇 점이야?", "청소를 잘하면 상점을 받을 수 있어?"],
    ).launch()
