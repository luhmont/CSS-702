#pip install streamlit pandas pymupdf sentence-transformers transformers spacy  
#pip install transformers torch     


import streamlit as st
import pandas as pd #for storing restaurant data
import pymupdf #for reading pdf files
from sentence_transformers import SentenceTransformer #for getting vectors from text
from pathlib import Path #for reading all pdfs in the folder
from transformers import GPT2Tokenizer, GPT2LMHeadModel #for tokenizing imput
import spacy
import pandas as pd
from transformers import pipeline
import torch
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity


sentencetransformermodel = SentenceTransformer("all-MiniLM-L6-v2")
restaurantdf = pd.DataFrame(columns=["Restaurant Name", "Keywords", "Embeddings"])
nlp = spacy.load("en_core_web_sm")

def load_gpt2():
    language_model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(language_model_name)
    nlpmodel = GPT2LMHeadModel.from_pretrained(language_model_name)
    return tokenizer, nlpmodel

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
         
def get_restaurantnames():
    pdf_search = Path(str(Path.cwd())).rglob("*.pdf") #gets all the pdf files in the current directory
    pdf_files = [str(file.absolute()) for file in pdf_search] #iterates over all the pdf files in the current directory and adds them to an array.
    return pdf_files

def get_restauranttext():
    pdf_search = Path(str(Path.cwd())).rglob("*.pdf") #gets all the pdf files in the current directory
    pdf_files = [str(file.absolute()) for file in pdf_search] #iterates over all the pdf files in the current directory and adds them to an array.
    pdf_filecontent = [pdf_reader(file) for file in pdf_files] #take the file names and extract the text content from them.
    return pdf_filecontent

def restaurant_datapull(text): #uses spaCy to extract Restaurant keywords, and add the title, keywords, and embeddings to a data frame.
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] #three labels for the types of extracted words
    doc = nlp(text.lower()) #lowercase all words
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation): #if word is in the stop words or is punctuation move on
            continue
        if(token.pos_ in pos_tag): #if not add to keywords
            result.append(token.text)
    return result

def get_embeddings(text):
    if isinstance(text, list):
        text = " ".join(text)
    embeddings = sentencetransformermodel.encode(text) #get embeddings of the text passed in the parameter
    return embeddings
    
#st.write("""# Restaurant Query Bot https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
#Hello *world!*""")

def get_Restaurant_info():
    restaurantdf
    restaurant_titles = get_restaurantnames()
    restaurant_texts = get_restauranttext()
    
    for name, text in zip(restaurant_titles, restaurant_texts):
        keywords = restaurant_datapull(text)
        embeddings = get_embeddings(keywords)
        restaurantdf.loc[len(restaurantdf)] = [name, keywords, embeddings]
    return restaurantdf

def find_best_match(prompt_embedding, restaurant_df):
    similarities = []
    for idx, embedding in enumerate(restaurant_df["Embeddings"]):
        similarity = cosine_similarity(
            [prompt_embedding], [embedding]  # Compare embeddings
        )[0][0]
        similarities.append((idx, similarity))
    
    # Sort restaurants by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0]  # Return the best match

def generate_response(prompt, best_restaurant):
    restaurant_name = best_restaurant['Restaurant Name']
    keywords = ", ".join(best_restaurant['Keywords'])
    response = (
        f"Here is my recommendation: '{restaurant_name}'. "
        f"It is popular for {keywords}. Enjoy your meal!"
    )
    return response


def main():
    
    get_Restaurant_info()
    
    st.title("Restaurant Query Bot") #page title
    with st.chat_message("assistant"): #gives name to bot and writes welcome message
        st.write("Hello, I'm your restaurant assistant, how can I help you?")
        
    tokenizer, model = load_gpt2() #chat history initialization
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Request a Restaurant"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
            # Display assistant response in chat message container
        
        if restaurantdf.empty:
            with st.chat_message("assistant"):
                st.markdown("Sorry, no restaurant data is available.")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, no restaurant data is available."})
            return

        prompt_keywordembeddings = get_embeddings(restaurant_datapull(prompt)) #get the vector of the keywords from the user's input
        best_match = find_best_match(prompt_keywordembeddings, restaurantdf)
        
        if not best_match:  # No match found
            with st.chat_message("assistant"):
                st.markdown("Sorry, no matching restaurants were found.")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, no matching restaurants were found."})
            return
        
        best_match_idx, _ = find_best_match(prompt_keywordembeddings, restaurantdf) #find best vector match for prompt using cosine similarity
        best_restaurant = restaurantdf.iloc[best_match_idx]

        response = generate_response(prompt, best_restaurant)
        
        '''gpt2_input = f"User asked about {prompt}. Recommend: {best_restaurant['Restaurant Name']} with keywords: {best_restaurant['Keywords']}." #have gpt return response
        input_ids = tokenizer.encode(gpt2_input, return_tensors="pt")
        
        if input_ids.size(1) > 1024:
            input_ids = input_ids[:, :1024]
            
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.eos_token_id
        output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                        do_sample=True,
                        temperature=0.9,
                        max_new_tokens=100,
                    )
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.eos_token_id
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)'''
            
        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    return

if __name__ == "__main__":
    main()