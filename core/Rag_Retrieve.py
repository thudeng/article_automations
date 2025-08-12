import os
import sys
import warnings
import tempfile
import shutil
from typing import List, Optional
from langchain.schema import Document
from typing import List, Dict, Tuple, Optional

# 设置环境变量和警告过滤
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizers并行问题
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class SafeRAGSystem:
    def __init__(self, api_key: str, use_openai_embeddings: bool = False):
        """
        初始化RAG系统

        Args:
            api_key: DeepSeek API密钥
            use_openai_embeddings: 是否使用OpenAI嵌入（需要OpenAI API Key）
        """
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.1
        )

        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.persist_directory = None
        self.use_openai_embeddings = use_openai_embeddings
        self.model_path = "..//model"

    def _get_embeddings(self):
        """获取嵌入模型"""
        if self.use_openai_embeddings:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings()
        else:
            try:
                # 尝试使用新版本的HuggingFace嵌入
                from langchain_huggingface import HuggingFaceEmbeddings
                print("Using langchain_huggingface.HuggingFaceEmbeddings")
            except ImportError:
                # 回退到旧版本
                from langchain_community.embeddings import HuggingFaceEmbeddings
                print("Using langchain_community.embeddings.HuggingFaceEmbeddings")

            return HuggingFaceEmbeddings(
                model_name=self.model_path,
                model_kwargs={
                    'device': 'cpu',  # 强制使用CPU
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 1  # 减少批处理大小以避免内存问题
                }
            )

    def get_relevant_segments(self, query: str, k: int = 3):
        """
        获取与查询最相关的文档片段

        Args:
            query: 查询文本
            k: 返回的片段数量

        Returns:
            list: 包含相关片段的列表，每个元素包含content和score
        """
        if not self.retriever:
            return []

        try:
            # 使用向量存储进行相似度搜索
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

            # 格式化返回结果
            relevant_segments = []
            for doc, score in docs_with_scores:
                segment = {
                    'content': doc.page_content,
                    'score': 1 - score,  # 转换为相似度分数，越高越相似
                    'metadata': doc.metadata
                }
                relevant_segments.append(segment)

            # 按相似度分数排序
            relevant_segments.sort(key=lambda x: x['score'], reverse=True)

            return relevant_segments

        except Exception as e:
            print(f"Error retrieving segments: {e}")
            return []

    def load_documents(self, folder_path: str) -> List:
        """从文件夹加载并分割所有.md格式的文档"""
        try:
            # 获取所有.md文件的路径
            md_files = [f for f in os.listdir(folder_path) if f.endswith('.md') or f.endswith('.txt')]
            if not md_files:
                print(f"No .md files found in {folder_path}")
                return []

            # 读取并加载文档内容
            docs = []
            for md_file in md_files:
                file_path = os.path.join(folder_path, md_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 创建Document对象，添加文件名作为metadata
                doc = Document(page_content=content, metadata={"file_name": md_file})
                docs.append(doc)

            # 文档分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # 减小块大小
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            print(f"Split into {len(splits)} chunks")

            return splits

        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

    def create_vectorstore(self, documents: List) -> bool:
        """创建向量存储"""
        if not documents:
            print("No documents to process")
            return False

        try:
            embeddings = self._get_embeddings()

            # 创建临时目录
            self.persist_directory = tempfile.mkdtemp()

            # 分批处理文档以避免内存问题
            batch_size = 10
            all_texts = []
            all_metadatas = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                for doc in batch:
                    all_texts.append(doc.page_content)
                    all_metadatas.append(doc.metadata)

            # 创建向量存储
            self.vectorstore = Chroma.from_texts(
                texts=all_texts,
                metadatas=all_metadatas,
                embedding=embeddings,
                persist_directory=self.persist_directory,
                collection_name="rag_docs"
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            print("Vector store created successfully")
            return True

        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False

    def create_rag_chain(self):
        """创建RAG链"""
        if not self.retriever:
            print("Retriever not initialized")
            return False

        try:
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

            print("RAG chain created successfully!")
            return True

        except Exception as e:
            print(f"Error creating RAG chain: {e}")
            return False

    def ask_question(self, question: str) -> str:
        """提问并获取答案"""
        if not self.rag_chain:
            return "RAG chain not initialized"
        try:
            response = self.rag_chain.invoke({"input": question})
            return response.get("answer", "No answer generated")
        except Exception as e:
            return f"Error processing question: {e}"

    def cleanup(self):
        """清理临时文件"""
        if self.persist_directory and os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print("Cleaned up temporary directory")
            except Exception as e:
                print(f"Error cleaning up: {e}")


def main():
    """主函数"""
    # 初始化RAG系统
    rag = SafeRAGSystem(
        api_key="sk-2f4ea37debc1420d82379d20963cba30",
        use_openai_embeddings=False  # 设置为True如果您有OpenAI API Key
    )

    try:
        # 加载文档
        documents = rag.load_documents("..//config//strategies")

        if not documents:
            print("Failed to load documents")
            return

        # 创建向量存储
        if not rag.create_vectorstore(documents):
            print("Failed to create vector store")
            return

        # 创建RAG链
        if not rag.create_rag_chain():
            print("Failed to create RAG chain")
            return

        # 测试问题
        test_questions = [
            "DeepBI是如何提曝光的",
        ]

        for question in test_questions:
            print(f"\nQ: {question}")
            answer = rag.ask_question(question)
            print(f"A: {answer}")
            print("-" * 50)
        for question in test_questions:
            print(f"\nQ: {question}")

            # 获取最相关的片段
            segments = rag.get_relevant_segments(question, k=3)

            if segments:
                print(f"找到 {len(segments)} 个相关片段:")
                for i, segment in enumerate(segments, 1):
                    print(f"\n片段 {i} (相似度: {segment['score']:.3f}):")
                    print(f"{segment['content'][:200]}...")  # 显示前200个字符
                    print("-" * 50)
            else:
                print("未找到相关片段")

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # 清理资源
        rag.cleanup()


if __name__ == "__main__":
    main()