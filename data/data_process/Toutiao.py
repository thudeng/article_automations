import pandas as pd
import requests
import schedule
import time
from datetime import datetime, timedelta
import requests
from datetime import datetime, timedelta
from io import BytesIO
# 获取日期范围函数
def get_date_range():
    today = datetime.today()
    # 获取过去7天的日期范围
    start_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    return start_date, end_date


# 用获取的 Cookies 下载数据
def download_data_with_cookies(cookies:list):
    cookie_string = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
    # 获取日期范围
    from_date, to_date = get_date_range()
    # 构建请求的 URL
    download_url_toutiao = f'https://mp.toutiao.com/mp/agw/statistic/v2/content/export_stat_trends?from={from_date}&to={to_date}&type=1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Cookie': cookie_string  # 使用提取并生成的 Cookie 字符串
    }
    # 发送 GET 请求并获取数据
    response = requests.get(download_url_toutiao, headers=headers)
    if response.status_code == 200:
        # 保存文件
        file_name = f"toutiao_data_{from_date}_to_{to_date}.txt"
        print(response.content)
        with open(file_name, 'wb') as file:
            file.write(response.content)
        # excel_data = BytesIO(response.content)
        # try:
        #     df = pd.read_excel(excel_data)
        #     print("成功读取Excel数据：")
        #     print(df.head())
        #     # 保存为Excel文件
        #     df.to_excel(file_name, index=False)
        #     print("数据已保存为toutiao_data.xlsx")
        # except Exception as e :
        #     print(f"保存失败：{e}")
    else:
        print(f"下载失败，状态码：{response.status_code}")
