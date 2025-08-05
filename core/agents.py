from openai import OpenAI

def abstract(theme,api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    with open("策略文档摘要.txt", "r", encoding='utf-8') as file:
        strategy = file.read().strip('\n')
    with open("问题-策略-摘要.txt", "r", encoding='utf-8') as file_2:
        preference = file_2.read().strip('\n')
    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": """你是一个精通亚马逊广告运营的专家，同时你还是一位高水平的作家，你写的摘要言简意赅，且不会出现歧义，逻辑性强，极具质量。"""},
                            {"role": "user", "content": f"""
                            #任务
                            请你围绕主题：{theme}写一篇摘要。
                            #需要学习的文档
                            其中，在写作摘要前，请你认真学习如下的文档：{strategy}
                            #需要参考的模板
                            你的摘要写作请参考如下的模板：{preference}
                            #摘要写作要求
                            1.不少于300字不大于700字
                            2.摘要中仅定性描述,且不分段
                            3.摘要中要说明具体的策略，策略名不要出错,且选择的策略数量不超过4个，所以你要挑重点。
                            4.逻辑性强，句意连贯。符合一篇摘要的特征。
                            5.特定的问题，不要带上四层流量机制！尽量不加入这个策略，就事论事"""}
                        ],
                        max_tokens=2000,
                        temperature=0.2
                    )
    return response.choices[0].message.content
def Analysis(theme,api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    with open("Analysis.txt", "r", encoding='utf-8') as file:
        preference = file.read().strip('\n')
    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": """你是一个精通亚马逊广告运营的专家，你对亚马逊广告相关问题的分析一刀见血，能够反应当前问题与趋势。"""},
                            {"role": "user", "content": f"""
                            #任务
                            请你围绕主题：{theme}写出你的分析。
                            #需要参考的模板
                            你的分析写作请参考如下的模板（格式）：{preference}
                            #你分析部分的其它写作要求
                            1.不少于100字不大于250字
                            2.你的分析要言简意赅，通俗易懂，分析重点要在运营策略上，而不是Listing这些
                            3.逻辑性强，句意连贯，不要分段。符合一篇高质量分析的特征。
                            4.只做定性描述，不要进行定量，不允许出现任何数字
                            5.只需要指出问题即可，不要介绍解决方法
                            6.你的回答中只给出你对主题的问题的分析即可
    """}
                        ],
                        max_tokens=2000,
                        temperature=0.2
                    )
    return response.choices[0].message.content
def Analysis(theme,api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    with open("Analysis.txt", "r", encoding='utf-8') as file:
        preference = file.read().strip('\n')
    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": """你是一个精通亚马逊广告运营的专家，你对亚马逊广告相关问题的分析一刀见血，能够反应当前问题与趋势。"""},
                            {"role": "user", "content": f"""
                            #任务
                            请你围绕主题：{theme}写出你的分析。
                            #需要参考的模板
                            你的分析写作请参考如下的模板（格式）：{preference}
                            #你分析部分的其它写作要求
                            1.不少于100字不大于250字
                            2.你的分析要言简意赅，通俗易懂，分析重点要在运营策略上，而不是Listing这些
                            3.逻辑性强，句意连贯，不要分段。符合一篇高质量分析的特征。
                            4.只做定性描述，不要进行定量，不允许出现任何数字
                            5.只需要指出问题即可，不要介绍解决方法
                            6.你的回答中只给出你对主题的问题的分析即可
    """}
                        ],
                        max_tokens=2000,
                        temperature=0.2
                    )
    return response.choices[0].message.content

def strategy_tradition(theme,api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    with open("strategy_tradition.txt", "r", encoding='utf-8') as file:
        preference = file.read().strip('\n')
    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": """你是一个精通亚马逊广告运营的专家，在如今AI时代，亚马逊广告运营模式将迎来转变。下面请你指出传统运营方法"""},
                            {"role": "user", "content": f"""
                            #任务
                            请你围绕主题：{theme}写出你的分析。
                            #需要参考的模板
                            你的传统运营方法写作请参考如下的模板（格式）：{preference}
                            #你传统运营方法部分的其它写作要求
                            1.不少于300字不大于800字
                            2.你的分析要言简意赅，通俗易懂。
                            3.逻辑性强，句意连贯，不要分段。
                            4.只做定性描述，不要进行定量，不允许出现任何数字
    """}
                        ],
                        max_tokens=2000,
                        temperature=0.2
                    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # analysis=strategy_tradition(theme="如何通过精准的ASIN竞价，迅速抢占亚马逊广告市场份额？",api_key="sk-2f4ea37debc1420d82379d20963cba30")
    # print(analysis)
    abstract_content=abstract(theme="如何通过精准的ASIN竞价，迅速抢占亚马逊广告市场份额？",api_key="sk-2f4ea37debc1420d82379d20963cba30")
    strategy_list=['自动加词策略','自动加ASIN策略','获取竞品ASIN与添加策略','重点词策略','控ACOS策略',
               '修改预算策略','基于库存-预算调整策略','四层流量机制','成单关键词与ASIN策略',
               '提曝光策略','控曝光策略']
    strategys=[]
    for strategy_choice in strategy_list:
        if strategy_choice in abstract_content:
            strategys.append(strategy_choice)
    print(strategys)
    print(abstract_content)