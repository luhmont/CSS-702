#pip install streamlit
#pip install pysimplegui NOT USED
#pip install --upgrade pymupdf
#pip install -U sentence-transformers
#python -m spacy download en_core_web_sm
#pip install transformers torch spacy

#there may be some dependency issues when trying to install spacy since there are issues with numpy 2.x and up, these lines seemed to solve them
#pip uninstall numpy
#pip install numpy==1.24.3
#pip install --upgrade pip
#pip install --upgrade contourpy numba pywavelets streamlit


       


import streamlit as st
import pandas as pd #for storing restuarant data
import pymupdf #for reading pdf files
from sentence_transformers import SentenceTransformer #for getting vectors from text
from pathlib import Path #for reading all pdfs in the folder
from transformers import GPT2Tokenizer, GPT2Model #for tokenizing imput
import spacy
import pandas as pd



model = SentenceTransformer("all-MiniLM-L6-v2")
restuarantdf = pd.DataFrame("Restuarant Name", "Keywords", "Embeddings")


def pdf_reader(file_path): #method for reading pdfs
    #open file
    file = pymupdf.open(file_path)
    
    #read the file
    #out = open("output.txt", "wb") #prints text to output file in the directory
    out = ""
    for page in file: #iterate the pages
        text = page.get_text() #get plain text
        out += text #write text of page
    #print(out)
        
    file.close()
    return out
         
def get_restuarantnames():
    pdf_search = Path(str(Path.cwd())).rglob("*.pdf") #gets all the pdf files in the current directory
    pdf_files = [str(file.absolute()) for file in pdf_search] #iterates over all the pdf files in the current directory and adds them to an array.
    return pdf_files

def get_restuaranttext():
    pdf_search = Path(str(Path.cwd())).rglob("*.pdf") #gets all the pdf files in the current directory
    pdf_files = [str(file.absolute()) for file in pdf_search] #iterates over all the pdf files in the current directory and adds them to an array.
    pdf_filecontent = [pdf_reader(file) for file in pdf_files] #take the file names and extract the text content from them.
    return pdf_filecontent

def restuarant_datapull(text): #uses spaCy to extract restuarant keywords, and add the title, keywords, and embeddings to a data frame.
    
    return

def get_embeddings(text):
    embeddings = model.encode(text) #get embeddings of the text passed in the parameter
    return embeddings
    
#st.write("""# Restuarant Query Bot https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
#Hello *world!*""")


def main():
    
    
    
    st.title("Restuarant Query Bot") #page title
    with st.chat_message("assistant"): #gives name to bot and writes welcome message
        st.write("Hello, I'm your restuarant assistant, how can I help you?")
        
    if "messages" not in st.session_state: #if no messages have been sent then initialize chat history
        st.session_state.messages = [] #variable for chat history
        
    for message in st.session_state.messages: #loop that iterates through the chat history to display the history in the message container.
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Say something"):
        st.chat_message("user").markdown(prompt) #display message in chat message container
        st.session_state.messages.append({"role":"user", "content": prompt}) #add message to chat history
        
        response = f"{prompt}"
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    return

if __name__ == "__main__":
    main()