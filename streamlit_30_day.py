#type directly into the terminal 
#this is my latest code taht works for resume 

#------mainline------#
#streamlit run streamlit_30_day.py  
#------mainline------#

import os
import json
import zipfile
import cassandra 
import streamlit as st

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import (
# pip install -r requirments.txt #install all requirements, if a requirements.txt file is present
# pip freeze > requirements.txt # after adding new requirements, this add them to the 
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from copy import deepcopy
from tempfile import NamedTemporaryFile


@st.cache_resource 
def create_datastax_connection():

  cloud_config= {'secure_connect_bundle': '/Users/nicoleschultz/Documents/secure-connect-reusme.zip'}

  with open('/Users/nicoleschultz/schultznma@gmail.com-token.json') as f:
      secrets = json.load(f)

  CLIENT_ID = secrets['clientId']
  CLIENT_SECRET = secrets['secret']

  # Create an auth provider with your credentials
  auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)

  # Finally, create the cluster and connect
  cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
  astra_session = cluster.connect()
  return astra_session


def main():
   # st.write("Hello, Streamlit!")
    
    st.set_page_config(
        page_title = "Chat with your PDF", 
        page_icon=" 🤖 ",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
        )
    st.header(' 🤖 Chat with your PDF')

    st.write("Trying to connect to database...")
    session = create_datastax_connection()
    st.write("Database connected!")


    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

##   session = create_datastax_connection()

##     #os.environ['OPEN_API_KEY'] = "API-key"

    os.environ['OPENAI_API_KEY'] = "INSERT KEY"
    llm = OpenAI(temperature=0)
    openai_embeddings = OpenAIEmbeddings()

#   table_name = "resume_table"
#   keyspace = "default_keyspace"

#   out_index_creator = VectorstoreIndexCreator(
#       vectorstore_cls = Cassandra,
#       embedding = openai_embeddings,
#       text_splitter = RecursiveCharacterTextSplitter(
#       chunk_size = 400,
#       chunk_overlap = 30),

#       vectorstore_kwargs={
#       'session': session,
#       'keyspace': keyspace,
#       'table_name': table_name}
#     )      



if __name__ == '__main__':
    main()

