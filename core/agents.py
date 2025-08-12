from openai import OpenAI, api_key
import pandas as pd
from typing import List, Dict
import os
class ArticleGenerationWorkflow:
    """文章生成工作流系统"""

    def __init__(self, api_key: str):
        """
        初始化工作流系统

        Args:
            api_key: OpenAI API密钥
            knowledge_base_path: 知识库路径，如果提供则初始化文档检索器
        """
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


    def generate_analysis(self, theme: str) -> str:
        """
        生成分析内容

        Args:
            theme: 主题

        Returns:
            str: 分析内容
        """
        try:
            # 读取策略理解文档
            Analysis_preference = pd.read_excel("../config/strategies/strategies_base/策略文档摘要.xlsx")
            Analysis_preference_dict = dict(zip(Analysis_preference['策略'], Analysis_preference['理解']))
            Analysis_understand = pd.read_excel("../config/strategies/strategies_base/问题-策略-摘要.xlsx")
            Analysis_understand_dict = dict(zip(Analysis_understand['情景'], Analysis_understand['策略方法']))
        except Exception as e:
            print(f"警告：无法读取策略文档: {e}")
            Analysis_preference_dict = {}
            Analysis_understand_dict = {}
        if not any(word in theme for word in ["新","初","机制"]):
                del Analysis_preference_dict["四层流量机制"]
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system","content": """你是一位精通亚马逊SP广告运营的专家，DeepBI是一款基于AI智能的广告投放助手，你将分析结合DeepBI的策略，给出亚马逊广告运营痛点的破解之道"""},
{"role": "user", "content": f"""
#痛点主题
{theme}
#DeepBI相关策略的理解
{Analysis_preference_dict}
#另一位专家对一些问题的思考供参考
{Analysis_understand_dict}
#任务
1.写一篇250字-450字的分析
2.请你学习DeepBI相关策略理解的文档，建立各策略之间的联系
3.逻辑性强，句意连贯，不要分段。
4.若分析涉及新品推广或“初期”阶段内容，需提到DeepBI的四层流量机制。
5.若分析不涉及新品推广或“初期”阶段内容，不要提及DeepBI的四层流量机制。
6.除非涉及到与"季节"、"周期"有关的话题，否则不要扯上SKU关闭这一项
7.就事论事，主要谈优化思路
                """}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        return response.choices[0].message.content

    def generate_abstract(self, theme: str, analysis: str, files_path: List[str] = None) -> str:
        """
        生成摘要

        Args:
            theme: 主题
            analysis: 分析内容
            files_path: 参考文件路径列表

        Returns:
            str: 摘要内容
        """
        content_dict = {}

        if files_path:
            for file_path in files_path:
                try:
                    full_path = f"../config/strategies/strategies_base/{file_path}"
                    with open(full_path, "r", encoding="utf-8") as file:
                        content_dict[file_path.strip(".md").strip(".txt")] = file.read()
                except Exception as e:
                    print(f"警告：无法读取文件 {file_path}: {e}")

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
{"role": "system","content": """你是一个精通亚马逊SP广告运营和BM25算法的专家且擅长写文章的摘要，DeepBI是一款基于AI智能的亚马逊广告投放系统。现在，你需要用一篇短小精悍的摘要文章去介绍DeepBI，你的摘要应容易通过BM25算法检索到给定主题相关的文章(为此供便利，DeepBI并未用该算法）"""},
{"role": "user", "content": f"""
#任务
写一篇约为600字，最多不超过1000字的摘要
#围绕的主题
{theme}
#结构和要求
1.主题的背景（15%）、传统SP广告运营的方法（25%）、DeepBI的方法（50%）、对比总结（10%）。
2.逻辑性强，句意连贯，按照第一点的要求分成4段进行论述，各模块之间要衔接自然。
3.你写的摘要尤其涉及到DeepBI策略的一定要把它的核心机制叙述完整，切不可只叙述其中的一部分而不完整（会造成断章取义，务必要避免）！
4.不要定量描述策略（如"7天ACOS<27%"等数量词），只需要定性描述：如用"近几天ACOS较好"）
5.针对该主题，DeepBI的策略分析如下：
{analysis}
#参考内容
有关DeepBI的分析，具体的策略补充如下：
{content_dict}
                """}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        return response.choices[0].message.content

    def generate_article(self, theme: str, abstract: str,analysis:str, files_path: List[str] = None,files_path_rag:str=None,use_rag:bool=True) -> str:
        """
        生成文章

        Args:
            theme: 主题
            abstract: 摘要
            files_path: 在不使用RAG的情况下，参考文件路径列表
            analysis:参考文章的分析
            files_path_rag:在使用RAG的情况下，参考文件路径列表
            use_rag:是否启用RAG检索
        Returns:
            str: 文章内容
        """
        content_dict = {}
        #可以选择两种检索方案：一种是基于RAG的片段检索，另外一种是基于分析的文档检索，并把全文给创作的Agent
        if use_rag:
            from Rag_Retrieve import SafeRAGSystem
            rag = SafeRAGSystem(api_key=self.api_key, use_openai_embeddings=False)
            try:
                # 加载RAG检索文档
                documents = rag.load_documents(files_path_rag)
                if not documents:
                    print("RAG检索路径错误，请检查".center(60, "*"))
                    return ""
                # 创建向量存储和RAG链
                if not rag.create_vectorstore(documents):
                    print("向量存储创建失败，请检查".center(60, "*"))
                    return ""
                if not rag.create_rag_chain():
                    print("RAG链创建失败，请检查".center(60, "*"))
                    return ""
                # 测试问题
                test_questions, segments_content = theme,[]
                # 获取最相关的片段,参数k表示返回的排名最前即最相关片段的数量
                segments = rag.get_relevant_segments(test_questions, k=5)
                if segments:
                    for i, segment in enumerate(segments, 1):
                        content_dict[f"参考片段{i}"] = segment["content"]
                else:
                    print("未找到相关片段".center(60,"*"))
            except Exception as e:
                print(f"Fatal error: {e}")
            finally:
                rag.cleanup()
        else:
            if files_path:
                for file_path in files_path:
                    try:
                        full_path = f"../config/strategies/{file_path}"
                        with open(full_path, "r", encoding="utf-8") as file:
                            content_dict[file_path.strip(".md").strip(".txt")] = file.read()
                    except Exception as e:
                        print(f"警告：无法读取文件 {file_path}: {e}")
        try:
            with open("../templates/Marketing_templates.md", "r", encoding="utf-8") as template:
                templates = template.read()
        except Exception as e:
            print(f"警告：无法读取模板文件: {e}")

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
{"role": "system","content": """你是一位亚马逊SP广告运营的商家也是一名知乎、头条、CSDN等图文平台的优秀创作者。DeepBI是一款基于AI智能的广告投放助手，你所在公司使用了DeepBI产品后实现了"由亏转盈"的"破局"和"改命"。现在你怀揣着感恩之心，根据所给的摘要去写一篇推介DeepBI产品的文章"""},
{"role": "user", "content": f"""
#主题
{theme}
#目的
根据所给的摘要生成一篇优秀的官方平台的文案用来以第三方的视角来推介DeepBI的产品。但是必须注意分寸，不能被博客平台识别为广告文。
#对文章的分析如下
{analysis}
#文章的摘要如下（即你写定文章的摘要）
{abstract}
#要求
1.字数在2200-2500字左右
2.涉及到策略内容的，不要定量描述策略（如"7天ACOS<27%"等数量词），只需要定性描述：如用"近几天ACOS较好"。
3.结构：前言（5%，内容要求：以五条围绕给定主题的问题开头，作用是为文章作为关键句来获取流量并直接承接下文）、背景（15%）、DeepBI解决的方案（45%）、优势（20%）、总结（15%）。给出合适的章节标题：要求醒目、吸引人而且多样化。结构尽量简单
4.格式：文章标题为1级标题，章节标题为2级标题，2级标题不要带有"，"、"："等，就用一句话简要概括。不要有其它的标题，可以用加粗来表示重点
5.语言风格：适用于各网络平台的文章风格
7.文章最后结束语应尽量拉近和读者的距离且应提到DeepBI，凸显DeepBI的优势和价值，并间接让读者产生试用DeepBI的想法。
9.文章的结尾标题统一用"总结"。
#创作结构模板(只是模板，你的文章必须严格按照摘要来，不许抄里面的任何内容！)
{templates}
#参考资料：
{content_dict}"""}],
            max_tokens=3000,
            temperature=0.5
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    API_KEY = "sk-2f4ea37debc1420d82379d20963cba30"
    KNOWLEDGE_BASE_PATH = "..\config\strategies"  # 知识库路径，如果不存在会自动跳过文档检索
    THEME = "新品推广如何通过亚马逊广告实现爆单"

    # 初始化工作流系统
    workflow = ArticleGenerationWorkflow(
        api_key=API_KEY,
    )
    print(workflow.generate_analysis("亚马逊广告运营进阶指南：提曝光 or 降曝光"))


#     print(Article_Creation
#           (theme="新品推广如何通过亚马逊广告实现爆单", api_key="sk-2f4ea37debc1420d82379d20963cba30",
#                            files_path=["DeepBI的四层流量机制.md","传统搜索关键词拓展方法的局限性.md","自动广告的宽泛困境与智能解法.md","Listing优秀搜索关键词的来源(传统和DeepBI的).md","DeepBI是如何精准找到流量的.md","市场竞争激烈，大词流量大，但是转化率低.md","传统新手亚马逊广告打法和DeepBI的新手打法以及其对比.md"],
#                            Abstract="""# 新品推广如何通过亚马逊广告实现爆单：传统运营与DeepBI智能策略对比
# ## 背景与挑战
# 亚马逊新品推广面临流量获取难、ACOS偏高、预算分配不合理等核心痛点。传统运营方式依赖人工经验，难以系统化解决这些问题，导致新品推广周期长、爆单概率低。随着亚马逊广告竞争加剧，卖家亟需更智能的解决方案实现精准流量获取与高效转化。
# ## 传统SP广告运营方法
# 传统新品推广主要依赖人工选词与ASIN投放，存在明显局限性。在关键词投放方面，运营者通常手动选取大词或竞品词，缺乏动态优化机制，导致ACOS居高不下。ASIN投放则局限于少量头部竞品，流量池狭窄且竞争激烈。预算分配往往固定不变，无法根据实时表现调整，造成预算浪费或优质流量获取不足。整个推广过程缺乏系统化分层机制，难以实现流量质量的阶梯式提升。
# ## DeepBI智能推广策略
# DeepBI通过四层流量机制与策略组合，构建了系统化的新品爆单解决方案。**四层流量机制**形成完整漏斗：探索层通过ASIN广告抢占竞品详情页和搜索结果页流量，实现初始数据积累；初筛层动态筛选潜力词与ASIN；精准层验证长期价值流量；放量层集中资源投放优质黑马词与ASIN。**核心策略组合**包括：自动加词策略挖掘真实搜索词构建动态词库，自动加ASIN策略扩展竞品流量池；提曝光与控曝光策略形成"探索-优化"闭环；成单关键词与重点词策略实现梯度化运营；基于库存的预算调整与动态修改预算策略保障投放持续性。DeepBI独特之处在于ASIN广告反哺关键词机制——通过竞品ASIN投放归因高转化关键词，解决新品期listing质量分低导致的关键词广告效果不佳问题。
# ## 策略对比与总结
# 相比传统方法，DeepBI展现出三大优势：在流量获取上，通过ASIN反哺关键词机制突破新品冷启动困境；在数据应用上，实现"采集-验证-放量"的螺旋上升式优化；在资源分配上，依托智能算法达成预算与流量的动态平衡。这种系统化策略组合使新品推广能够快速积累有效数据，精准识别高价值流量，最终实现爆单目标。对于亚马逊卖家而言，采用DeepBI智能系统可显著缩短新品推广周期，提高爆单成功率。"""
# ))
