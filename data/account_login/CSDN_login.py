import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from data.data_process.Toutiao import download_data_with_cookies
from data.data_process.JueJin import get_article_stats

def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--incognito")  # 开启隐身模式，禁用缓存
    chrome_options.add_argument("--disable-application-cache")  # 禁用应用缓存
    chrome_options.add_argument("--disable-cache")  # 禁用缓存
    chrome_options.add_argument("--disk-cache-size=0")  # 设置磁盘缓存为 0
    chrome_options.add_argument("--start-maximized")  # 最大化窗口
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    # 使用 Service 类来指定 ChromeDriver 路径
    service = Service(ChromeDriverManager().install())
    # 启动 Chrome 浏览器
    driver = webdriver.Chrome(service=service,options=chrome_options)  # 使用 service 和 options 参数
    return driver

def get_cookies_CSDN(user_name:str,password:str):# 模拟登录并获取 cookies
    # 初始化 Chrome 浏览器
    driver = init_driver()
    # 打开登录页面
    driver.get("https://juejin.cn/")  # 这里替换成你实际登录的页面
    time.sleep(5)
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='juejin']/div[1]/div/header/div/nav/ul/ul/li[3]/div/button"))
    )
    driver.execute_script("arguments[0].scrollIntoView();", login_button)
    login_button.click()
    # 等待登录框加载并显示
    time.sleep(4)
    driver.find_element(By.XPATH, '//*[@id="juejin"]/div[2]/div[3]/form/div[2]/div[1]/div[1]/div[2]/span').click()
    time.sleep(1)
    # 模拟登录过程：根据页面的实际元素定位填写用户名和密码，并提交
    username_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH,'//*[@id="juejin"]/div[2]/div[3]/form/div[2]/div[1]/div[1]/div[1]/div[1]/div[1]/input'))
    )

    password_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH,'//*[@id="juejin"]/div[2]/div[3]/form/div[2]/div[1]/div[1]/div[1]/div[1]/div[3]/input'))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", password_input)
    username_input.send_keys(user_name)
    password_input.send_keys(password)
    time.sleep(3)
    driver.find_element(By.XPATH,'//*[@id="juejin"]/div[2]/div[3]/form/div[2]/div[1]/div[1]/div[1]/div[2]/button[2]').click()
    time.sleep(30)  # 提交登录表单
    # 获取登录后的 cookies
    cookies = driver.get_cookies()
    get_article_stats(cookies=cookies)
    # 关闭浏览器
    driver.quit()