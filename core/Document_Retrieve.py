from rank_bm25 import BM25Okapi
import jieba
import numpy as np
import os
import glob
from typing import List, Tuple, Optional, Dict


class DocumentRetriever:
    """智能文档检索系统"""

    def __init__(self, knowledge_base_path: str, file_extensions: List[str] = None):
        """
        初始化文档检索器

        Args:
            knowledge_base_path: 知识库文件夹路径
            file_extensions: 支持的文件扩展名，默认为 ['.md', '.txt']
        """
        self.knowledge_base_path = knowledge_base_path
        self.file_extensions = file_extensions or ['.md', '.txt']

        # 存储文档数据
        self.documents = []
        self.file_names = []
        self.tokenized_documents = []
        self.tokenized_titles = []

        # BM25 模型
        self.bm25_content = None
        self.bm25_title = None

        # 语义关系库
        self.semantic_relations = self._init_semantic_relations()

        # 停用词
        self.stop_words = {
            '是', '一种', '的', '了', '在', '和', '与', '或', '但', '而', '因为', '所以',
            '这', '那', '些', '个', '把', '被', '让', '使', '得', '着', '过', '来', '去',
            '上', '下', '中', '里', '外', '前', '后', '左', '右', '有', '会', '能', '可以',
            '就', '都', '要', '说', '对', '为', '如何', '什么', '怎么', '时候', '应该',
            '或者', 'ROI', 'ACOS', '亚马逊', '广告', '为什么', '运营', '往往',"首先","然后","最后","总结","进阶","指南"
        }

        self.unmeaningful_chars = {
            "为什么", "呢", "的", "地", "得", "如何", "实现", "还是",
            '什么', '时候', '怎么', '？', '呢'
        }

        # 加载文档
        self._load_documents()

    def _init_semantic_relations(self) -> Dict[Tuple[str, str], float]:
        """初始化语义关系库"""
        semantic_relations = {
            # 曝光相关
            ('提高', '曝光'): 2.0,
            ('广告', '优化'): 1.8,
            ('曝光', '优化'): 1.7,
            ('曝光', '预算'): 1.5,
            ('关键词', '匹配'): 1.5,
            ('曝光', '策略'): 1.6,
            ('竞价', '优化'): 1.4,
            ('广告', '数据分析'): 1.3,
            ('目标', '受众'): 1.5,
            ('广告', '投放'): 1.2,
            ('曝光', '点击'): 1.2,
            ('流量', '提升'): 1.5,
            ('智能', '投放'): 1.6,
            ('产品', '推广'): 1.4,
            ('广告', '成效'): 1.3,
            ('曝光', '分析'): 1.3,
            ('广告', '自动化'): 1.2,
            ('投放', '优化'): 1.5,
            ('曝光', '提升'): 2.0,
            ('广告', '监控'): 1.4,
            ('营销', '策略'): 1.5,

            # AI相关
            ('AI', '广告'): 1.5,
            ('AI', '策略'): 1.7,
            ('AI', '投放'): 1.6,
            ('AI', '流量'): 1.5,
            ('AI', '自动'): 1.6,
            ('AI', '算法'): 1.8,
            ('AI', '预算'): 1.5,
            ('AI', '人工'): 1.6,
            ('AI', '实时'): 1.4,
            ('AI', '自动化'): 1.7,
            ('AI', '特点'): 1.5,
            ('AI', '优势'): 1.8,
            ('AI', '优点'): 1.6,
            ('人工', '缺点'): 1.7,
            ('人工', '传统'): 1.7,
            ('AI', 'DeepBI'): 1.0,
        }
        return semantic_relations

    def _load_documents(self):
        """加载知识库文档"""
        if not os.path.exists(self.knowledge_base_path):
            raise FileNotFoundError(f"知识库文件夹不存在: {self.knowledge_base_path}")

        # 获取所有支持的文档路径
        document_paths = []
        for ext in self.file_extensions:
            pattern = os.path.join(self.knowledge_base_path, f"*{ext}")
            document_paths.extend(glob.glob(pattern))

        if not document_paths:
            extensions_str = ', '.join(self.file_extensions)
            raise FileNotFoundError(f"在知识库中未找到支持的文件类型: {extensions_str}")

        print(f"找到 {len(document_paths)} 个文件:")

        # 读取文档内容
        for document_path in document_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.documents.append(content)

                    file_name = os.path.basename(document_path)
                    self.file_names.append(file_name)
                    print(f"✓ 已加载: {file_name}")
            except Exception as e:
                print(f"读取文件出错 {os.path.basename(document_path)}: {e}")
                self.documents.append("")
                self.file_names.append(os.path.basename(document_path))

        # 分词处理
        self.tokenized_documents = [self._tokenize(doc) for doc in self.documents]
        self.tokenized_titles = [self._tokenize(name) for name in self.file_names]

        # 初始化BM25模型
        self.bm25_content = BM25Okapi(self.tokenized_documents, k1=1.5, b=0.75)
        self.bm25_title = BM25Okapi(self.tokenized_titles, k1=1.5, b=0.75)

        print(f"文档库初始化完成，共加载 {len(self.documents)} 个文档")

    def _tokenize(self, text: str) -> List[str]:
        """文本分词并过滤停用词"""
        tokens = list(jieba.cut(text))
        meaningful_chars = {'广告', '投放', '曝光', '流量', '价格', '预算', '关键词', '优势'}
        filtered_tokens = []

        for token in tokens:
            token = token.strip()
            if len(token) > 1 or token in meaningful_chars:
                if token not in self.stop_words and token not in filtered_tokens:
                    filtered_tokens.append(token)

        return filtered_tokens

    def _tokenize_filtered(self, tokens: List[str]) -> List[str]:
        """进一步过滤无意义的词"""
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if len(token) > 1 and token not in self.unmeaningful_chars:
                filtered_tokens.append(token)
        return filtered_tokens

    def _semantic_boost(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """基于语义关系计算额外分数"""
        boost = 0
        # for q_token in query_tokens:
        #     for d_token in doc_tokens:
        #         if (q_token, d_token) in self.semantic_relations:
        #             boost += self.semantic_relations[(q_token, d_token)]
        #         elif (d_token, q_token) in self.semantic_relations:
        #             boost += self.semantic_relations[(d_token, q_token)]
        #语义增强目前弃用，所以无论如何均返回0
        return boost

    def _exact_match_score(self, q_tokens: List[str], d_tokens: List[str]) -> int:
        """计算完全匹配分数"""
        matches = sum(1 for token in q_tokens if token in d_tokens)
        return matches if q_tokens else 0

    def _qa_pattern_score(self, query: str, doc_tokens: List[str]) -> float:
        """问答模式识别分数"""
        score = 0
        query_filtered = self._tokenize_filtered(self._tokenize(query))

        if any(word in query for word in ['什么时候', '如何', '怎么', '为什么', '？', '呢']):
            for query_token in query_filtered:
                if query_token in doc_tokens:
                    score += 1
        return score

    def search(self, query: str, top_k: int = 5, show_details: bool = True) -> List[Tuple[str, float, int]]:
        """
        搜索最相关的文档

        Args:
            query: 查询文本
            top_k: 返回前k个最相关的文档
            show_details: 是否显示详细的计算过程

        Returns:
            List[Tuple[文件名, 最终得分, 文档索引]]
        """
        if not self.documents:
            raise ValueError("请先加载文档")

        # 分词查询
        tokenized_query = self._tokenize(query)

        bm25_scores_content = self.bm25_content.get_scores(tokenized_query)
        bm25_scores_title = self.bm25_title.get_scores(tokenized_query)

        final_scores = []

        for i, filename in enumerate(self.file_names):
            # BM25分数
            bm25_content = bm25_scores_content[i]
            bm25_title = bm25_scores_title[i]

            # 完全匹配分数
            exact_content = self._exact_match_score(tokenized_query, self.tokenized_documents[i])
            exact_title = self._exact_match_score(tokenized_query, self.tokenized_titles[i])

            # 问答模式分数
            qa_content = self._qa_pattern_score(query, self.tokenized_documents[i])
            qa_title = self._qa_pattern_score(query, self.tokenized_titles[i])

            # 语义增强分数
            semantic_content = self._semantic_boost(tokenized_query, self.tokenized_documents[i])
            semantic_title = self._semantic_boost(tokenized_query, self.tokenized_titles[i])

            # 加权融合 (标题权重更高)
            final_score_content = bm25_content + exact_content + qa_content + semantic_content * 0.1
            final_score_title = (bm25_title * 15 + exact_title * 2 + qa_title * 10 + semantic_title * 0.1)

            # 最终得分（标题权重80%，内容权重20%）
            final_score = final_score_title * 0.8 + final_score_content * 0.2
            final_scores.append(final_score)

            if show_details:
                pass
                # print(f"[{filename}]:")
                # print(f"  BM25: 内容={bm25_content:.3f}, 标题={bm25_title:.3f}")
                # print(f"  完全匹配: 内容={exact_content}, 标题={exact_title}")
                # print(f"  问答模式: 内容={qa_content:.3f}, 标题={qa_title:.3f}")
                # print(f"  语义增强: 内容={semantic_content:.3f}, 标题={semantic_title:.3f}")
                # print(f"  最终得分: {final_score:.6f}")
                # print()

        # 排序并返回结果
        scored_docs = [(final_scores[i], self.file_names[i], i) for i in range(len(self.file_names))]
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        result = [(filename, score, index) for score, filename, index in scored_docs[:top_k]]

        if show_details:
            print("")
            print(f"\n🎯 前{top_k}个最相关文档:")
            for rank, (filename, score, index) in enumerate(result, 1):
                print(f"{rank}. [{filename}] - 得分: {score:.6f}")
                if rank == 1 and self.documents[index]:
                    # 显示最相关文档的预览
                    doc_preview = self.documents[index][:200].replace('\n', ' ').strip()
                    print(f"   📝 内容预览: {doc_preview}...")

        return result

    def get_best_match(self, query: str, show_details: bool = True) -> Optional[Tuple[str, str, float]]:
        """
        获取最佳匹配的文档

        Returns:
            Tuple[文件名, 文档内容, 得分] 或 None
        """
        results = self.search(query, top_k=1, show_details=show_details)
        if results:
            filename, score, index = results[0]
            return filename, self.documents[index], score
        return None

    def add_semantic_relation(self, word1: str, word2: str, weight: float):
        """添加新的语义关系"""
        self.semantic_relations[(word1, word2)] = weight

    def update_knowledge_base(self, new_path: Optional[str] = None):
        """更新知识库"""
        if new_path:
            self.knowledge_base_path = new_path

        # 重新加载文档
        self.documents = []
        self.file_names = []
        self._load_documents()


# 使用示例
if __name__ == "__main__":
    # 初始化检索器
    retriever = DocumentRetriever("..\config\strategies")
    # 示例查询
    query = """在亚马逊广告运营中，曝光量的调控需要根据广告阶段和目标动态调整策略组合。对于初期探索阶段，DeepBI的提曝光策略通过智能竞价调整快速获取曝光，配合自动加词策略和自动加ASIN策略形成关键词和竞品ASIN的拓展闭环，这种双轨机制既能挖掘高潜力长尾词，又能通过竞品ASIN精准截流。当广告进入稳定期后，成单关键词与ASIN策略会对历史转化词进行激进提价，而重点词策略则对近期高转化词实施更大力度的溢价培养，形成阶梯式的价值挖掘体系。针对过度曝光导致的成本问题，控曝光策略通过动态降价将曝光拉回合理区间，而控ACOS策略则专门压制高花费低转化的劣质流量，二者协同实现成本精细管控。预算管理方面，修改预算策略根据ACOS表现动态调配预算资源，结合基于库存的预算调整策略防止断货风险，构成完整的预算防御体系。对于市场竞争激烈的大词，可通过自动加词策略持续获取用户真实搜索词，配合搜索词竞品ASIN策略拓展次级竞品库，实现流量来源的多元化布局。整个优化过程形成"探索-筛选-放大-控制"的闭环逻辑，各策略通过数据反馈机制相互联动，既保证流量获取效率，又维持健康的广告成本结构。
"""
    # 搜索相关文档
    results = retriever.search(query, top_k=46)
    # 获取最佳匹配

    best_match = retriever.get_best_match(query, show_details=False)
    if best_match:
        filename, content, score = best_match
        print(f"\n最佳匹配: {filename} (得分: {score:.6f})")

    # 添加自定义语义关系
    retriever.add_semantic_relation("策略", "优化", 1.5)