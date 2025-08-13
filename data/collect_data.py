from typing import Any,Dict,List
# from data.account_login.Toutiao_login import get_cookies_toutiao
# from data.account_login.JueJin_login import get_cookies_juejin
import json


def json_encode_rollback(json_file,last_four_digits,platform) -> dict[Any, Any] | None | Any:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if platform == "TouTiao":
        for key,value in data[0]["TouTiao"].items():
            if value["phone_last_four_digits"] == last_four_digits:
              return value
        return {}
    if platform == "JueJin":
        for key, value in data[1]["JueJin"].items():
          if value["phone_last_four_digits"] == last_four_digits:
            return value
        return {}


print(json_encode_rollback("..//config//strategies//strategies_base//account.json", "1379", "JueJin"))





# class DataStatistics:
#     def __init__(self,json_path:str=None):
#         self.json_path = json_path
# 
#     def _TouTiao(self):
