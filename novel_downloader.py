
# まずはデータ抽出
import os
import time
import datetime
import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup
import streamlit as st

def make_bs_obj(url):
    """
    BeautifulSoupObjectを作成
    """
    html = urlopen(url)
    #logger.debug('access {} ...'.format(url))
    st.info("access {} ...".format(url))

    return BeautifulSoup(html,"html.parser")


#なろうはこちらを利用
def get_main_text_narou(bs_obj):
    """
    各話のコンテンツをスクレイピング
    """
    text = ""
    text_htmls = bs_obj.findAll("div",{"id":"novel_honbun"})[0].findAll("p")

    for text_html in text_htmls:
        text = text + text_html.get_text() + "\n\n"

    return text

# ノベプラはこちらを利用
def get_main_text_nobera(bs_obj):
    """
    各話のコンテンツをスクレイピング
    """
    text = ""
#    text_htmls = bs_obj.findAll("div",{"id":"novel_honbun"})[0].findAll("p") #これはなろう小説用
    text_htmls = bs_obj.findAll("div",{"class":"content"})[0].findAll("p") #これはノベプラ用

#    print(text_htmls)
#    print(bs_obj.findAll("div",{"class":"content"})[0].findAll("p"))

    for text_html in text_htmls:
        text = text + text_html.get_text() + "\n\n"

    return text


#なろうはこちらを呼び出す
@st.cache(allow_output_mutation=True, max_entries=10, ttl=3600,suppress_st_warning=True)
def narou_download(url_list):

    """
    データ格納メイン処理（一度取得したら/drive/MyDrive/novelsに入るので後は不要）
    """
    # --------------------------------------
    # 作品ページのURLを指定（なずな作品。n0464em,n3008cxは非小説）
    #url_list = ["https://ncode.syosetu.com/n7708cv/","https://ncode.syosetu.com/n8738es/","https://ncode.syosetu.com//n0464em/"]
    # --------------------------------------
    # 各作品について処理
    stories = []
    for url in url_list:
        bs_obj = make_bs_obj(url)
        time.sleep(3)

        url_list = ["https://ncode.syosetu.com" + a_bs_obj.find("a").attrs["href"] for a_bs_obj in bs_obj.findAll("dl", {"class": "novel_sublist2"})]
        date_list = bs_obj.findAll("dt",{"class":"long_update"})
        novel_title = bs_obj.find("p",{"class":"novel_title"}).get_text()
        for s in r'\/*?"<>:|':
            novel_title = novel_title.replace(s, '')

        # 各話の本文情報を取得
        for j in range(len(url_list)):
            url = url_list[j]
            bs_obj = make_bs_obj(url)
            time.sleep(3)

            stories.append({
                "No": j+1,
                "title": bs_obj.find("p", {"class": "novel_subtitle"}).get_text(),
                "url": url,
                "date": date_list[j].get_text(),
                "text": get_main_text_narou(bs_obj),
                })

    return stories

#ノベプラはこちらを呼び出す
@st.cache(allow_output_mutation=True, max_entries=10, ttl=3600,suppress_st_warning=True)
def nobera_download(url_list):

    """
    メイン処理
    """
    # --------------------------------------
    # 作品ページのURLを指定
    #url_list = ["https://novelup.plus/story/990043834"]
    #url_list = ["https://novelup.plus/story/692793647"]
    # --------------------------------------
    # 各作品について処理
    stories = []
    for url in url_list:
        bs_obj = make_bs_obj(url)
        time.sleep(3)

        url_list = [a_bs_obj.find("a").attrs["href"] for a_bs_obj in bs_obj.findAll("div", {"class": "episode_link episode_show_visited"})]
        date_list = bs_obj.findAll("dt",{"class":"long_update"})
        novel_title = bs_obj.find("div",{"class":"novel_title"}).get_text()
        for s in r'\/*?"<>:|':
            novel_title = novel_title.replace(s, "")

        novel_title = novel_title.replace("\\n", "")


        # 各話の本文情報を取得
        for j in range(len(url_list)):
            url = url_list[j]
            bs_obj = make_bs_obj(url)
            time.sleep(3)

            stories.append({
                "No": j+1,
                "title": bs_obj.find("div", {"class": "novel_title"}).get_text(),
                "url": url,
    #            "date": date_list[j].get_text(),
                "date": "",
                "text": get_main_text_nobera(bs_obj),
                })

    return stories

