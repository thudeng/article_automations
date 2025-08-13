import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from data.data_process.Toutiao import download_data_with_cookies
from selenium.webdriver.common.action_chains import ActionChains
# 清除浏览器缓存并初始化浏览器
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--incognito")  # 开启隐身模式，禁用缓存
    chrome_options.add_argument("--disable-application-cache")  # 禁用应用缓存
    chrome_options.add_argument("--disable-cache")  # 禁用缓存
    chrome_options.add_argument("--disk-cache-size=0")  # 设置磁盘缓存为 0
    chrome_options.add_argument("--start-maximized")  # 最大化窗口
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_argument("--headless")  # 启用无头模式
    # 使用 Service 类来指定 ChromeDriver 路径
    service = Service(ChromeDriverManager().install())
    # 启动 Chrome 浏览器
    driver = webdriver.Chrome(service=service,options=chrome_options)  # 使用 service 和 options 参数
    return driver

def get_cookies_toutiao(user_name:str,password:str):# 模拟登录并获取 cookies
    # 初始化 Chrome 浏览器
    driver = init_driver()
    # 打开登录页面
    driver.get("https://www.toutiao.com/")  # 这里替换成你实际登录的页面
    time.sleep(5)
    login_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div[3]/div[2]/a"))
    )
    driver.execute_script("arguments[0].scrollIntoView();", login_button)
    login_button.click()
    # 等待登录框加载并显示
    time.sleep(4)
    driver.find_element(By.XPATH, '//*[@id="login_modal_ele"]/div/article/article/div[2]/div/ul/li[4]/i').click()
    time.sleep(5)
    driver.find_element(By.XPATH, '//*[@id="login_modal_ele"]/div/article/article/div[1]/div[1]/div[2]/article/div[4]/span[1]').click()
    time.sleep(5)
    # 模拟登录过程：根据页面的实际元素定位填写用户名和密码，并提交
    username_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH,'//*[@id="login_modal_ele"]/div/article/article/div[1]/div[1]/div[2]/article/div[1]/div/input'))
    )

    password_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH,'//*[@id="login_modal_ele"]/div/article/article/div[1]/div[1]/div[2]/article/div[2]/div/div/input'))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", password_input)
    #输入账号和密码
    username_input.send_keys(user_name)
    password_input.send_keys(password)
    time.sleep(5)
    driver.find_element(By.XPATH,'//*[@id="login_modal_ele"]/div/article/article/div[1]/div[1]/div[2]/article/div[5]/button').click()
    # 等待页面跳转或登录成功
    time.sleep(3)
    try:
        action = ActionChains(driver)
        target = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//*[@id='root']/div/div[5]/div[2]/div[1]/div/a/div/span"))
        )
        action.move_to_element(target).click().perform()
    except Exception as e:
        print(f"Error clicking: {e}")
    # 等待页面跳转或登录成功
    time.sleep(5)
    # 获取登录后的 cookies
    cookies = driver.get_cookies()
    download_data_with_cookies(cookies)
    # 关闭浏览器
    driver.quit()

get_cookies_toutiao(user_name="17701371379",password="Deepbi888")