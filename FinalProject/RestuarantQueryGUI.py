#pip install streamlit
#pip install pysimplegui NOT USED
#pip install --upgrade pymupdf
#pip install -U sentence-transformers

import streamlit as st
import pandas as pd #for storing restuarant data
import pymupdf #for reading pdf files
from sentence_transformers import SentenceTransformer #for getting vectors from text
from pathlib import Path #for reading all pdfs in the folder

model = SentenceTransformer("all-MiniLM-L6-v2")

def pdf_reader(file_path): #test method for reading pdfs
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
         
def get_restuarants():
    pdf_search = Path(str(Path.cwd())).rglob("*.pdf") #gets all the pdf files in the current directory
    pdf_files = [str(file.absolute()) for file in pdf_search] #iterates over all the pdf files in the current directory and adds them to an array.

    pdf_filecontent = [pdf_reader(file) for file in pdf_files] #take the file names and extract the text content from them.

    menuembeddings = model.encode(pdf_filecontent)
    
#st.write("""# Restuarant Query Bot
#Hello *world!*""")
