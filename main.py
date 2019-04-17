from selenium import webdriver
import time

browser = webdriver.Chrome(executable_path="C:\\Users\\guodong\\Desktop\\chromedriver.exe")
browser.get('https://open.egame.qq.com/material/moment')

# time.sleep(2)
frame = browser.find_element_by_id('_egame_login_frame_qq_')
browser.switch_to.frame(frame)
# browser.switch_to.frame('_egame_login_frame_qq_')
browser.find_element_by_id('switcher_plogin').click()
browser.find_element_by_id('u').send_keys('2728567538')
browser.find_element_by_id('p').send_keys('Qq6781287GUO')
browser.find_element_by_id('login_button').click()
time.sleep(2)  # 休眠一定时间，等待其加载相应文件

browser.switch_to.parent_frame()
browser.find_element_by_link_text('高光时刻').click()
time.sleep(10)
all_videos_list = browser.find_element_by_xpath('//*[@id="my-app"]/div[1]/div/section/div[2]/div[1]/div[3]/ul')
print(browser.page_source)
