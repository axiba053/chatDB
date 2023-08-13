# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# 公众号/视频号/B站 ：三强的小屋
# Description：上传文本建立向量数据库
"""
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
import streamlit as st

class CreateDB:
    def __init__(self,dbName,File,chunk_size=500,chunk_overlap=50):#,API
        # self.API=API
        self.dbName=dbName
        self.file=File
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        st.warning('待处理文件为：'+self.file.split("/")[-1])

    def split_file(self):
        loader = UnstructuredFileLoader(self.file)
        documents = loader.load()
        st.warning(f'文件中共有{len(documents[0].page_content)}个字符串')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs=text_splitter.split_documents(documents)
        st.warning(f'共拆分为{len(docs)}个子数据')
        return docs

    def store_qdrand(self):
        docs=self.split_file()
        st.warning(f'向量数据库名称为：{self.dbName}')
        embeddings = OpenAIEmbeddings()#openai_api_key=self.API
        Qdrant.from_documents(
            docs, embeddings,
            path="./db",
            collection_name=self.dbName,
        )
        st.success('已成功新建数据库！')

def main():
    load_dotenv()
    File = 'data/test.docx'
    dbName='test'
    db=CreateDB(dbName,File)
    db.store_qdrand()

if __name__ == '__main__':
    main()

