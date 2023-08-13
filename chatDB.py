# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# 公众号/视频号/B站 ：三强的小屋
# Description：根据提问查询数据库，将查询结果输入chatgpt输出答案
"""
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import os
from createDB import CreateDB
import pathlib

def pre_brief():
    # 标题 图标
    image = Image.open('./image/AIgirl.png')
    st.set_page_config(page_title='客服', layout='wide', page_icon=image)
    st.header('咨询客服💁')
    #左边栏
    with st.sidebar:
        st.image(image, caption="", width=50)
        # st.markdown("---")
        st.markdown("## 客服使用说明")
        st.markdown("1. 上传资料文件，创建/更新数据库")
        st.markdown("2. 选择数据库提问")
        st.markdown("---")
        st.markdown("## 作者简介")
        st.markdown("大家好，这里是:blue[《三强的小屋》]！")
        st.markdown("欢迎在:green[B站]或者:green[微信视频号]关注，了解更多精彩内容!")
        image = Image.open("./image/B.jpg")
        st.image(image, caption="", width=250)

def suf_brief():
    st.markdown("---")
    st.write("您好,欢迎关注我的**B站账号、微信视频号及公众号**👏")
    st.write("作为知识的搬运工，我会持续更新AI相关技术及使用🧐")
    st.write("未来AI会像电脑一样普及，早用晚用不如现在就用！让我们一起探索AI的魅力吧🤔")
    st.write("️非常感谢您的关注与支持！🙏")
    st.write("")
    # 图片展示
    image1 = Image.open("./image/shipinhao.jpg")
    image2 = Image.open("./image/dingyuehao.jpg")
    image3 = Image.open("./image/wexin2.jpg")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image1, caption="微信视频号", width=200)
    with col2:
        st.image(image2, caption="微信订阅号", width=200)
    with col3:
        st.image(image3, caption="微信交流", width=200)

def get_chain(DbName):
    embeddings = OpenAIEmbeddings()
    client = QdrantClient(path="./db", prefer_grpc=True)
    vector_store = Qdrant(
        client=client,
        collection_name=DbName,
        embeddings=embeddings
    )
    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return chain

def ui():
    t1, t2 = st.tabs(['创建数据库', '选择数据库提问'])
    with t1:
        File=st.file_uploader('上传pdf/docx/txt/csv/md文件',
                              type=['pdf','docx','txt','csv','md'],
                         )
        if File:
            newFile = './tmp/' + File.name
            with open(pathlib.Path(newFile), 'wb') as f:
                f.write(File.read())
        dbName = st.text_input('输入新建/更新数据库名称',disabled= not File)
        sub = st.button('提交',disabled= not (File and dbName))
        if sub:
            db = CreateDB(dbName, newFile)
            db.store_qdrand()
    with t2:
        #读取数据库名称列表
        dbFiles=os.listdir('./db/collection')
        dbNames = tuple([name for name in dbFiles if not name.startswith('.')])
        DbName=st.selectbox('选择数据库：',dbNames)
        chain = get_chain(DbName)
        question = st.text_input('提问：',disabled= not DbName)
        if question:
            st.write(f'问：{question}')
            answer = chain.run(question)
            st.write(f'答：{answer}')
    st.write("")

def main():
    load_dotenv()
    pre_brief()
    ui()
    suf_brief()

if __name__ == '__main__':
    main()

