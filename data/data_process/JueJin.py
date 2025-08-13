import requests
import json
import pandas as pd


def get_article_stats(cookies,uuid,aid,item_id):
    # 设置API的URL
    url = f"https://api.juejin.cn/content_api/v1/author_center/data/trend?aid={aid}&uuid={uuid}&spider=0"

    # 设置负载数据
    payload = {
        "date_range": 1,
        "item_id": f"{item_id}",
        "item_type": 1,
        "datas": [
            "incr_article_display",
            "incr_article_view",
            "incr_article_digg",
            "incr_article_comment",
            "incr_article_collect"
        ]
    }

    # 设置请求头
    headers = {
        "Content-Type": "application/json"
    }

    # 设置cookie
    cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
    print(cookie_string)

    # 设置cookie到请求头中
    headers["Cookie"] = cookie_string

    # 发送POST请求
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # 检查是否成功响应
    if response.status_code == 200:
        # 解析返回的JSON数据
        data = response.json()
        print(data)

        # 提取日期和对应的阅读数、展现数
        dates = []
        views = []
        displays = []

        # 提取具体的字段
        for view, display in zip(data['data']['datas']['incr_article_view'],
                                 data['data']['datas']['incr_article_display']):
            dates.append(view['date'])
            views.append(view['cnt'])
            displays.append(display['cnt'])

        # 创建DataFrame
        df = pd.DataFrame({
            "Date": dates,
            "Article View": views,
            "Article Display": displays
        })
        print(df)

        # 保存到Excel
        df.to_excel("article_stats.xlsx", index=False)
        print("数据已成功保存到 article_stats.xlsx")
    else:
        print(f"请求失败，状态码：{response.status_code}")

# 使用示例：
# cookies = [{'name': 'cookie_name', 'value': 'cookie_value'}, ...]
# get_article_stats(cookies)
