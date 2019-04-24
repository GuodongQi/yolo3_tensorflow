# coding=utf-8
import os
from os import path
from multiprocessing import pool, process

from selenium import webdriver
import time
import urllib.request

out_path = "F:\\wzry"
total_tast = 10


def init():
    browser = webdriver.Chrome(executable_path="C:\\Users\\guodong\\Desktop\\chromedriver.exe")
    browser.get('https://open.egame.qq.com/material/moment')

    # time.sleep(2)
    frame = browser.find_element_by_id('_egame_login_frame_qq_')
    browser.switch_to.frame(frame)
    # browser.switch_to.frame('_egame_login_frame_qq_')
    browser.find_element_by_id('switcher_plogin').click()
    browser.find_element_by_id('u').send_keys('2728567538')
    browser.find_element_by_id('p').send_keys('Qq6781287GUO')
    time.sleep(2)  # 休眠一定时间，等待其加载相应文件
    browser.find_element_by_id('login_button').click()
    time.sleep(5)
    browser.switch_to.parent_frame()
    browser.find_element_by_link_text('高光时刻').click()
    time.sleep(5)
    total_pages, next_page = browser.find_element_by_class_name('manage-page').find_elements_by_tag_name('a')[
                             -2:]
    return browser, int(total_pages.text), next_page


def ger_from_per_page(browser):
    all_videos = browser.find_element_by_xpath(
        '//*[@id="my-app"]/div[1]/div/section/div[2]/div[1]/div[3]/ul').find_elements_by_tag_name('li')

    video_num = len(all_videos)
    all_titles = []
    all_links = []
    for i in range(video_num):
        try:
            all_links.append(
                all_videos[i].find_element_by_class_name('mange-list-bot').find_element_by_tag_name('a').get_attribute(
                    'href'))
            all_titles.append(all_videos[i].find_element_by_class_name('mange-list-bt').text.split('/')[1])
        except:
            print('{} download url not found'.format(
                all_videos[i].find_element_by_class_name('mange-list-bt').text.split('/')[1]))
    return all_titles, all_links


def download_video(all_titles, all_links, task_id, page_num, total_pages):
    for j in range(task_id, len(all_links), total_tast):
        title = all_titles[j]
        hero = title[:-2]
        kill_num = title[-2:]
        storage_dir = path.join(out_path, kill_num, hero)
        if not path.exists(storage_dir):
            os.makedirs(storage_dir)
        video_name = path.join(storage_dir, all_links[j].split('/')[-2] + '.mp4')
        if not path.exists(video_name):
            urllib.request.urlretrieve(all_links[j], video_name)
            print('page:{}/{}, video:{}/{}, storage in {} '.format(page_num + 1, total_pages, j + 1, len(all_links),
                                                                   video_name))
        else:
            print('page:{}/{}, video:{}/{}, {} has existed'.format(page_num + 1, total_pages, j + 1, len(all_links),
                                                                   video_name))


def main():
    browser, total_pages, next_page = init()
    for i in range(total_pages - 1):
        all_titles, all_links = ger_from_per_page(browser)
        p = pool.Pool(total_tast)
        for j in range(total_tast):
            p.apply_async(download_video, (all_titles, all_links, j, i, total_pages))
        p.close()
        p.join()
        print('page{} has been done'.format(i + 1))
        next_page.click()
        # time.sleep(2)
    return


if __name__ == '__main__':
    main()
