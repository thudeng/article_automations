import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
from openai import OpenAI
import hashlib


@dataclass
class DocumentChunk:
    """文档片段数据类"""
    id: str
    content: str
    source_file: str
    chunk_index: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = None


class ChineseTextSplitter:
    """中文文本分块器"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source_file: str = "") -> List[DocumentChunk]:
        """
        将文本分割成chunks

        Args:
            text: 要分割的文本
            source_file: 源文件名

        Returns:
            List[DocumentChunk]: 文档片段列表
        """
        if not text.strip():
            return []

        chunks = []
        text_length = len(text)

        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for paragraph in paragraphs:
            # 如果单个段落就超过chunk_size，需要进一步分割
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunk_id = self._generate_chunk_id(source_file, chunk_index)
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=current_chunk.strip(),
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    ))
                    chunk_index += 1
                    current_chunk = ""

                # 分割长段落
                sub_chunks = self._split_long_paragraph(paragraph)
                for sub_chunk in sub_chunks:
                    chunk_id = self._generate_chunk_id(source_file, chunk_index)
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=sub_chunk,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=text.find(sub_chunk),
                        end_pos=text.find(sub_chunk) + len(sub_chunk)
                    ))
                    chunk_index += 1
            else:
                # 检查是否需要创建新chunk
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    # 保存当前chunk
                    chunk_id = self._generate_chunk_id(source_file, chunk_index)
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=current_chunk.strip(),
                        source_file=source_file,
                        chunk_index=chunk_index,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk)
                    ))
                    chunk_index += 1

                    # 处理重叠
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                        current_chunk = overlap_text + "\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_start = text.find(paragraph) - len(overlap_text) if self.chunk_overlap > 0 else text.find(
                        paragraph)
                else:
                    # 添加到当前chunk
                    if current_chunk:
                        current_chunk += "\n" + paragraph
                    else:
                        current_chunk = paragraph
                        current_start = text.find(paragraph)

        # 保存最后一个chunk
        if current_chunk.strip():
            chunk_id = self._generate_chunk_id(source_file, chunk_index)
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                source_file=source_file,
                chunk_index=chunk_index,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk)
            ))

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """分割长段落"""
        chunks = []
        sentences = [s.strip() for s in paragraph.split('。') if s.strip()]

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk + "。")
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += "。" + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk + "。" if not current_chunk.endswith('。') else current_chunk)

        return chunks

    def _generate_chunk_id(self, source_file: str, chunk_index: int) -> str:
        """生成chunk ID"""
        return f"{source_file}_{chunk_index}_{hashlib.md5(f'{source_file}_{chunk_index}'.encode()).hexdigest()[:8]}"


class DocumentVectorStore:
    """文档向量存储"""

    def __init__(self, api_key: str = None, model_name: str = "text-embedding-3-small", use_embeddings: bool = True):
        self.api_key = api_key
        self.model_name = model_name
        self.use_embeddings = use_embeddings

        # 根据API密钥判断使用哪个服务
        if api_key and use_embeddings:
            if api_key.startswith("sk-") and "deepseek" not in api_key:
                # OpenAI API
                self.client = OpenAI(api_key=api_key)
                self.api_type = "openai"
            else:
                # DeepSeek API
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                self.api_type = "deepseek"
                # DeepSeek暂不支持embedding，改用纯BM25
                self.use_embeddings = False
                print("⚠️ DeepSeek API暂不支持embedding，将使用纯BM25检索")
        else:
            self.client = None
            self.api_type = None
            self.use_embeddings = False
            print("ℹ️ 未启用向量检索，将使用纯BM25检索")

        # 存储chunks和向量
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []
        self.chunk_index: Dict[str, int] = {}  # chunk_id -> index

        # BM25索引
        self.bm25_index = None
        self.tokenized_chunks = []

    def add_documents(self, chunks: List[DocumentChunk]):
        """添加文档片段"""
        print(f"正在处理 {len(chunks)} 个文档片段...")

        # 生成embeddings（如果启用）
        new_embeddings = []

        if self.use_embeddings and self.client:
            batch_size = 50

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]

                print(f"正在生成embeddings: {i + 1}-{min(i + batch_size, len(chunks))} / {len(chunks)}")

                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch_texts
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    new_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"生成embeddings失败: {e}")
                    print("⚠️ 将禁用向量检索，使用纯BM25检索")
                    self.use_embeddings = False
                    new_embeddings = []
                    break
        else:
            print("ℹ️ 跳过向量embedding生成，使用BM25检索")

        # 更新存储
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            self.chunk_index[chunk.id] = start_idx + i

        self.chunks.extend(chunks)
        self.embeddings.extend(new_embeddings)

        # 构建BM25索引
        self._build_bm25_index()

        print(f"✅ 成功添加 {len(chunks)} 个文档片段")

    def _build_bm25_index(self):
        """构建BM25索引"""
        print("正在构建BM25索引...")

        # 分词
        self.tokenized_chunks = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            self.tokenized_chunks.append(tokens)

        # 构建BM25
        if self.tokenized_chunks:
            self.bm25_index = BM25Okapi(self.tokenized_chunks, k1=1.5, b=0.75)

        print("✅ BM25索引构建完成")

    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        # 停用词
        stop_words = {
            '是', '一种', '的', '了', '在', '和', '与', '或', '但', '而', '因为', '所以',
            '这', '那', '些', '个', '把', '被', '让', '使', '得', '着', '过', '来', '去',
            '上', '下', '中', '里', '外', '前', '后', '左', '右', '有', '会', '能', '可以',
            '就', '都', '要', '说', '对', '为', '如何', '什么', '怎么', '时候', '应该'
        }

        tokens = list(jieba.cut(text))
        filtered_tokens = []

        for token in tokens:
            token = token.strip()
            if len(token) > 1 and token not in stop_words:
                filtered_tokens.append(token)

        return filtered_tokens

    def similarity_search(self, query: str, k: int = 5, search_type: str = "hybrid") -> List[
        Tuple[DocumentChunk, float]]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            search_type: 搜索类型 ("vector", "bm25", "hybrid")

        Returns:
            List[Tuple[DocumentChunk, float]]: (文档片段, 相似度分数)
        """
        if not self.chunks:
            return []

        # 如果不支持向量搜索，强制使用BM25
        if not self.use_embeddings and search_type in ["vector", "hybrid"]:
            print(f"⚠️ 向量搜索不可用，使用BM25搜索")
            search_type = "bm25"

        if search_type == "vector":
            return self._vector_search(query, k)
        elif search_type == "bm25":
            return self._bm25_search(query, k)
        else:  # hybrid
            return self._hybrid_search(query, k)

    def _vector_search(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """向量相似性搜索"""
        if not self.use_embeddings or not self.client:
            print("⚠️ 向量搜索不可用，请使用BM25搜索")
            return []

        try:
            # 获取查询向量
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[query]
            )
            query_embedding = response.data[0].embedding

            # 计算余弦相似度
            query_np = np.array(query_embedding)
            embeddings_np = np.array(self.embeddings)

            # 计算点积
            similarities = np.dot(embeddings_np, query_np)
            # 归一化
            norms = np.linalg.norm(embeddings_np, axis=1) * np.linalg.norm(query_np)
            similarities = similarities / norms

            # 获取top-k结果
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                chunk = self.chunks[idx]
                score = similarities[idx]
                results.append((chunk, float(score)))

            return results

        except Exception as e:
            print(f"向量搜索失败: {e}")
            print("⚠️ 回退到BM25搜索")
            return self._bm25_search(query, k)

    def _bm25_search(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """BM25搜索"""
        if not self.bm25_index:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)

        # 获取top-k结果
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有得分的结果
                chunk = self.chunks[idx]
                score = scores[idx]
                results.append((chunk, float(score)))

        return results

    def _hybrid_search(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """混合搜索（向量 + BM25）"""
        # 如果向量搜索不可用，直接使用BM25
        if not self.use_embeddings:
            return self._bm25_search(query, k)

        # 获取两种搜索的结果
        vector_results = self._vector_search(query, k * 2)
        bm25_results = self._bm25_search(query, k * 2)

        # 如果向量搜索失败，只使用BM25结果
        if not vector_results:
            return bm25_results

        # 归一化分数并合并
        chunk_scores = {}

        # 处理向量搜索结果
        vector_scores = [score for _, score in vector_results]
        if vector_scores:
            max_vector_score = max(vector_scores)
            min_vector_score = min(vector_scores)
            score_range = max_vector_score - min_vector_score

            for chunk, score in vector_results:
                # 归一化到0-1
                normalized_score = (score - min_vector_score) / score_range if score_range > 0 else 0
                chunk_scores[chunk.id] = chunk_scores.get(chunk.id, 0) + normalized_score * 0.6  # 向量权重60%

        # 处理BM25搜索结果
        bm25_scores = [score for _, score in bm25_results]
        if bm25_scores:
            max_bm25_score = max(bm25_scores)
            min_bm25_score = min(bm25_scores)
            score_range = max_bm25_score - min_bm25_score

            for chunk, score in bm25_results:
                # 归一化到0-1
                normalized_score = (score - min_bm25_score) / score_range if score_range > 0 else 0
                chunk_scores[chunk.id] = chunk_scores.get(chunk.id, 0) + normalized_score * 0.4  # BM25权重40%

        # 获取chunks映射
        chunk_map = {chunk.id: chunk for chunk in self.chunks}

        # 排序并返回top-k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for chunk_id, score in sorted_chunks:
            if chunk_id in chunk_map:
                results.append((chunk_map[chunk_id], score))

        return results

    def save_index(self, filepath: str):
        """保存索引到文件"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'chunk_index': self.chunk_index,
            'tokenized_chunks': self.tokenized_chunks
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"索引已保存到: {filepath}")

    def load_index(self, filepath: str):
        """从文件加载索引"""
        if not os.path.exists(filepath):
            print(f"索引文件不存在: {filepath}")
            return False

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.chunk_index = data['chunk_index']
            self.tokenized_chunks = data.get('tokenized_chunks', [])

            # 重建BM25索引
            if self.tokenized_chunks:
                self.bm25_index = BM25Okapi(self.tokenized_chunks, k1=1.5, b=0.75)

            print(f"✅ 索引已从 {filepath} 加载，包含 {len(self.chunks)} 个文档片段")
            return True

        except Exception as e:
            print(f"加载索引失败: {e}")
            return False


class RAGSystem:
    """RAG系统主类"""

    def __init__(self, api_key: str = None, knowledge_base_path: str = "", index_cache_path: str = None,
                 use_embeddings: bool = True):
        self.api_key = api_key
        self.knowledge_base_path = knowledge_base_path
        self.index_cache_path = index_cache_path or "rag_index.pkl"

        # 初始化组件
        self.text_splitter = ChineseTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vector_store = DocumentVectorStore(api_key, use_embeddings=use_embeddings)

        # 如果有API密钥，初始化客户端
        if api_key:
            if api_key.startswith("sk-") and "deepseek" not in api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.client = None

        # 文件扩展名
        self.supported_extensions = ['.md', '.txt']

    def build_index(self, force_rebuild: bool = False):
        """构建文档索引"""
        # 检查是否需要重建索引
        if not force_rebuild and os.path.exists(self.index_cache_path):
            if self.vector_store.load_index(self.index_cache_path):
                return

        print(f"开始构建文档索引...")
        print(f"扫描路径: {self.knowledge_base_path}")

        if not os.path.exists(self.knowledge_base_path):
            print(f"错误: 知识库路径不存在: {self.knowledge_base_path}")
            return

        # 收集所有文档文件
        all_chunks = []
        file_count = 0

        for root, dirs, files in os.walk(self.knowledge_base_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    file_path = os.path.join(root, file)
                    print(f"处理文件: {file}")

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if content.strip():
                            chunks = self.text_splitter.split_text(content, file)
                            all_chunks.extend(chunks)
                            file_count += 1
                            print(f"  ✅ 生成 {len(chunks)} 个片段")

                    except Exception as e:
                        print(f"  ❌ 处理失败: {e}")

        print(f"\n总计处理 {file_count} 个文件，生成 {len(all_chunks)} 个文档片段")

        if all_chunks:
            # 添加到向量存储
            self.vector_store.add_documents(all_chunks)

            # 保存索引
            self.vector_store.save_index(self.index_cache_path)
            print("✅ 索引构建完成")
        else:
            print("⚠️ 未找到有效的文档内容")

    def retrieve_documents(self, query: str, top_k: int = 5, search_type: str = "hybrid") -> List[
        Tuple[DocumentChunk, float]]:
        """
        根据查询检索相关文档片段

        Args:
            query: 查询文本（如摘要）
            top_k: 返回结果数量
            search_type: 搜索类型 ("vector", "bm25", "hybrid")

        Returns:
            List[Tuple[DocumentChunk, float]]: 相关文档片段和分数
        """
        print(f"\n🔍 检索查询: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"搜索类型: {search_type}, 返回数量: {top_k}")

        results = self.vector_store.similarity_search(query, top_k, search_type)

        print(f"✅ 找到 {len(results)} 个相关文档片段:")
        for i, (chunk, score) in enumerate(results, 1):
            print(f"{i}. [{chunk.source_file}] 得分: {score:.4f}")
            print(f"   内容预览: {chunk.content[:100]}...")
            print()

        return results

    def retrieve_by_files(self, query: str, target_files: List[str], top_k: int = 5, search_type: str = "hybrid") -> \
    List[Tuple[DocumentChunk, float]]:
        """
        在指定文件中检索相关内容

        Args:
            query: 查询文本
            target_files: 目标文件列表
            top_k: 每个文件返回的片段数量
            search_type: 搜索类型

        Returns:
            List[Tuple[DocumentChunk, float]]: 相关文档片段和分数
        """
        print(f"\n🎯 在指定文件中检索:")
        for file in target_files:
            print(f"  - {file}")

        # 获取所有结果
        all_results = self.vector_store.similarity_search(query, len(self.vector_store.chunks), search_type)

        # 按文件过滤
        file_results = []
        file_counts = {}

        for chunk, score in all_results:
            if chunk.source_file in target_files:
                if file_counts.get(chunk.source_file, 0) < top_k:
                    file_results.append((chunk, score))
                    file_counts[chunk.source_file] = file_counts.get(chunk.source_file, 0) + 1

        print(f"✅ 在目标文件中找到 {len(file_results)} 个相关片段")
        return file_results

    def generate_context(self, query: str, retrieved_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """
        根据检索到的文档片段生成上下文

        Args:
            query: 原始查询
            retrieved_chunks: 检索到的文档片段

        Returns:
            str: 生成的上下文
        """
        if not retrieved_chunks:
            return "未找到相关文档内容。"

        # 按文件分组
        file_groups = {}
        for chunk, score in retrieved_chunks:
            if chunk.source_file not in file_groups:
                file_groups[chunk.source_file] = []
            file_groups[chunk.source_file].append((chunk, score))

        # 构建上下文
        context_parts = []
        context_parts.append(
            f"基于查询「{query[:100]}{'...' if len(query) > 100 else ''}」，从知识库中检索到以下相关内容：\n")

        for file_name, chunks in file_groups.items():
            context_parts.append(f"## 来源文件: {file_name}\n")

            for i, (chunk, score) in enumerate(chunks, 1):
                context_parts.append(f"### 片段 {i} (相关度: {score:.3f})")
                context_parts.append(f"{chunk.content}\n")

        return "\n".join(context_parts)

    def get_file_statistics(self) -> Dict[str, Any]:
        """获取文档库统计信息"""
        if not self.vector_store.chunks:
            return {"总文档片段数": 0, "文件数": 0, "平均片段长度": 0}

        file_counts = {}
        total_length = 0

        for chunk in self.vector_store.chunks:
            file_counts[chunk.source_file] = file_counts.get(chunk.source_file, 0) + 1
            total_length += len(chunk.content)

        return {
            "总文档片段数": len(self.vector_store.chunks),
            "文件数": len(file_counts),
            "平均片段长度": total_length // len(self.vector_store.chunks),
            "文件分布": file_counts
        }


# 使用示例
if __name__ == "__main__":
    # 配置参数
    API_KEY = "sk-2f4ea37debc1420d82379d20963cba30"  # DeepSeek API密钥
    KNOWLEDGE_BASE_PATH = "..\config\strategies"

    # 初始化RAG系统（针对DeepSeek API使用纯BM25）
    rag = RAGSystem(
        api_key=API_KEY,
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        index_cache_path="rag_index.pkl",
        use_embeddings=False  # 对于DeepSeek API，设为False使用纯BM25
    )

    # 构建索引（首次运行需要）
    print("🚀 构建文档索引...")
    rag.build_index(force_rebuild=False)  # 设为True强制重建

    # 示例查询（摘要）
    abstract_query = """
    新品推广如何通过亚马逊广告实现爆单的核心策略包括四层流量机制与智能算法优化。
    传统方法依赖人工选词投放，存在ACOS偏高、流量获取困难等问题。
    DeepBI通过自动加词策略、自动加ASIN策略以及预算智能分配实现精准流量获取。
    """

    # 全局检索（使用BM25）
    print("\n" + "=" * 60)
    print("1. 全局文档检索（BM25）")
    print("=" * 60)

    results = rag.retrieve_documents(
        query=abstract_query,
        top_k=8,
        search_type="bm25"  # 使用BM25搜索
    )

    # 生成检索上下文
    if results:
        context = rag.generate_context(abstract_query, results)
        print("\n📄 生成的上下文:")
        print("-" * 40)
        print(context[:1000] + "..." if len(context) > 1000 else context)

    # 指定文件检索
    print("\n" + "=" * 60)
    print("2. 指定文件检索")
    print("=" * 60)

    target_files = ["自动加词策略.md", "自动加ASIN策略.txt", "四层流量机制.md"]
    file_results = rag.retrieve_by_files(
        query=abstract_query,
        target_files=target_files,
        top_k=3,
        search_type="bm25"
    )

    # 获取统计信息
    stats = rag.get_file_statistics()
    print(f"\n📊 文档库统计:")
    print("-" * 40)
    for key, value in stats.items():
        if key != "文件分布":
            print(f"{key}: {value}")

    if "文件分布" in stats and stats["文件分布"]:
        print(f"\n📁 文件分布:")
        for file, count in stats["文件分布"].items():
            print(f"  {file}: {count} 个片段")

    # 演示不同搜索类型的效果
    print("\n" + "=" * 60)
    print("3. 搜索类型对比")
    print("=" * 60)

    test_query = "自动加词策略如何优化ACOS"

    bm25_results = rag.retrieve_documents(test_query, top_k=3, search_type="bm25")
    print(f"\n🔍 BM25搜索结果 ({len(bm25_results)} 个):")
    for i, (chunk, score) in enumerate(bm25_results, 1):
        print(f"{i}. [{chunk.source_file}] 得分: {score:.4f}")
        print(f"   内容: {chunk.content[:100]}...")

    # 如果启用了向量搜索，也显示对比
    if rag.vector_store.use_embeddings:
        try:
            vector_results = rag.retrieve_documents(test_query, top_k=3, search_type="vector")
            print(f"\n🧠 向量搜索结果 ({len(vector_results)} 个):")
            for i, (chunk, score) in enumerate(vector_results, 1):
                print(f"{i}. [{chunk.source_file}] 得分: {score:.4f}")
                print(f"   内容: {chunk.content[:100]}...")
        except:
            print("\n⚠️ 向量搜索不可用")