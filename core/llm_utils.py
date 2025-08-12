from openai import OpenAI
import pandas as pd
from typing import List, Dict, Tuple, Optional
import os
from Document_Retrieve import DocumentRetriever
from agents import ArticleGenerationWorkflow
from Rag_Retrieve import SafeRAGSystem


def save_results(results: Dict[str, str], output_dir: str = "..//deepseek_articles"):
    """
    保存生成结果到文件

    Args:
        results: 生成结果字典
        output_dir: 输出目录
    """
    import datetime

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%m-%d")
    file_path = os.path.join(output_dir, timestamp)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    theme_safe = "".join(c for c in results['theme'][:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()

    # 保存各个文件
    files_saved = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 保存分析
    analysis_file = os.path.join(file_path, f"{timestamp}_{theme_safe}_分析.txt")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(results['analysis'])
    files_saved.append(analysis_file)

    # 保存摘要
    abstract_file = os.path.join(file_path, f"{timestamp}_{theme_safe}_摘要.txt")
    with open(abstract_file, 'w', encoding='utf-8') as f:
        f.write(results['abstract'])
    files_saved.append(abstract_file)

    # 保存文章
    article_file = os.path.join(file_path, f"{timestamp}_{theme_safe}_文章.md")
    with open(article_file, 'w', encoding='utf-8') as f:
        f.write(results['article'])
    files_saved.append(article_file)

    print(f"\n💾 结果已保存到以下文件:")
    for file_path in files_saved:
        print(f"  - {file_path}")


class DocumentCreation:
    """智能文档生成系统"""
    def __init__(self, api_key: str, knowledge_base_path: str = None,use_rag:bool=True):
        """
        初始化工作流系统

        Args:
            api_key: OpenAI API密钥
            knowledge_base_path: 知识库路径，如果提供则初始化文档检索器
            file_path_rag:如果进行RAG检索，检索文件的路径
        """
        self.api_key = api_key
        self.use_rag = use_rag
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        if knowledge_base_path and os.path.exists(knowledge_base_path):
            self.retriever = DocumentRetriever(knowledge_base_path)
            print("测试通过")
        else:
            print("检索文档失败".center(60,"*"))
        self.article_generation = ArticleGenerationWorkflow(api_key=api_key)

    def auto_select_reference_docs(self, abstract: str, top_k: int = 10) -> List[str]:
        """
        根据摘要自动选择参考文档

        Args:
            abstract: 摘要内容
            top_k: 返回前k个相关文档

        Returns:
            List[str]: 相关文档文件名列表
        """
        if not self.retriever:
            print("警告：文档检索器未初始化，返回空列表")
            return []

        print(f"\n🔍 正在根据摘要内容自动选择参考文档...")
        relevant_docs = self.retriever.search(abstract, top_k=top_k, show_details=True)
        files_name=[]
        print(f"\n✅ 自动选择了 {len(relevant_docs)} 个相关文档:")
        for rank, (filename, score, index) in enumerate(relevant_docs, 1):
            files_name.append(filename)

        return files_name

    def generate_full_workflow(self, theme: str,file_path_rag:str="../config/strategies",manual_files: List[str] = None,
                                top_k_docs: int = 8) -> Dict[str, str]:
        """
        执行完整的文章生成工作流

        Args:
            theme: 文章主题
            manual_files: 手动指定的参考文件列表
            file_path_rag: RAG检索文档的路径
            top_k_docs: 在不启用RAG检索的模式下，自动选择文档的数量

        Returns:
            Dict[str, str]: 包含各步骤结果的字典
        """
        print(f"\n🚀 开始文章生成工作流")
        print(f"主题: {theme}")
        print("=" * 60)

        # 步骤1: 生成分析
        print("\n📊 步骤1: 生成策略分析...")
        analysis = self.article_generation.generate_analysis(theme)
        print("✅ 分析生成完成")
        print(analysis)
        # 步骤2: 生成摘要
        print("\n📝 步骤2: 生成摘要...")
        abstract = self.article_generation.generate_abstract(theme, analysis, manual_files)
        # 步骤3: 生成最终文章
        print(f"\n✍️  步骤3: 生成最终文章...")
        reference_docs,article_content=[],""
        if self.use_rag:
            article_content = self.article_generation.generate_article(theme=theme, abstract=abstract, analysis=analysis,
                                                               files_path_rag=file_path_rag,use_rag=True)
            print(f"\n🎉 工作流执行完成！")
            return {"theme": theme, "article": article_content, "abstract": abstract, "analysis": analysis,
                    "reference_docs": reference_docs}
        else:
            if self.retriever:
                reference_docs = self.auto_select_reference_docs(theme, top_k_docs)
                article_content = self.article_generation.generate_article(theme=theme, abstract=abstract,analysis=analysis,use_rag=False)
            print(f"\n🎉 工作流执行完成！")
            return {"theme":theme,"article":article_content,"abstract":abstract,"analysis":analysis}


# 使用示例
if __name__ == "__main__":
    # 配置参数
    API_KEY = "sk-2f4ea37debc1420d82379d20963cba30"
    KNOWLEDGE_BASE_PATH = "..//config//strategies"  # 知识库路径，如果不存在会自动跳过文档检索
    THEME = "亚马逊广告进阶指南：如何快速打开新品的流量"

    # 初始化工作流系统
    workflow = DocumentCreation(
        api_key=API_KEY,
        knowledge_base_path=KNOWLEDGE_BASE_PATH,  # 如果路径不存在，会自动禁用文档检索
        use_rag = True
    )

    # 执行完整工作流
    results = workflow.generate_full_workflow(
        theme=THEME,
        top_k_docs=8  # 自动选择8个最相关文档

    )

    # 打印结果摘要
    print(f"\n" + "=" * 60)
    print("📋 生成结果摘要:")
    print("=" * 60)
    print(f"分析长度: {len(results['analysis'])} 字符")
    print(f"摘要长度: {len(results['abstract'])} 字符")
    print(f"文章长度: {len(results['article'])} 字符")

    # 保存结果到文件
    save_results(results)

    # 如果需要，可以打印具体内容
    print(f"\n📊 生成的分析:")
    print("-" * 40)
    print(results['analysis'])

    print(f"\n📝 生成的摘要:")
    print("-" * 40)
    print(results['abstract'])

    print(f"\n📚 选择的参考文档:")
    print("-" * 40)