# -*- coding: utf-8 -*-
#%% Imports

import re 
import numpy as np
import torch

from resources import abbreviations

#%% Funtions 
def split_text_into_sentences(text):
    text = re.sub("[!?]",".",text)
    
    while text.find("..") > -1:
        text = text.replace("..",".")
    
    # "."-exceptions
    exceptions_list = abbreviations
    
    month_list = ["januar", "februar", "marts", "april", "maj", "juni",
             "juli", "august", "september", "oktober", "november", "december"]
    
    for i in range(0,10):
        for month in month_list:
            date = f"{i}. {month}"
            replacement = f"{i}%% {month}"
            text = text.replace(date,replacement)
    
    for exception in exceptions_list:
        text = text.replace(exception[0],exception[1])
       
    pkt_instances = []
    pkt_instances = pkt_instances + re.findall("[0-9]\. pkt", text)
    pkt_instances = pkt_instances + re.findall("[0-9]\. og [0-9]", text)
    pkt_instances = pkt_instances + re.findall("[0-9]\., [0-9]", text)
    
    number_instances = re.findall("[0-9]\. ", text)
    pkt_instances = pkt_instances + number_instances
    
    pkt_replacements = []
    for instance in pkt_instances:
        pkt_replacements.append(instance.replace('.','%%'))
    
    for i in range(0,len(pkt_replacements)):
        text = text.replace(pkt_instances[i],pkt_replacements[i])
        
    sentence_end = re.findall("%% [A-Z]", text) + re.findall("%% §", text)
    sentence_end_dot = [x.replace("%%",".") for x in sentence_end] 
    
    for i in range(0,len(sentence_end)):
        text = text.replace(sentence_end[i], sentence_end_dot[i])
    
    text = text.replace('jf.','jf%%')
    
    
    # reversing "."-exceptions after split
    sentence_list = []
    while text.find('.') > 0:
        sentence_text = text[0:text.find('.')+1]
        
        for exception in exceptions_list:
            sentence_text = sentence_text.replace(exception[1],exception[0])
        
        for i in range(0,10):
            for month in month_list:
                date = f"{i}%% {month}"
                replacement = f"{i}. {month}"
                sentence_text = sentence_text.replace(date,replacement)
        
        for i in range(0,len(pkt_replacements)):
            sentence_text = sentence_text.replace(pkt_replacements[i],pkt_instances[i])
        
        text = text[text.find('.')+1:len(text)]
        
        sentence_list.append(sentence_text)
        
    if text.find('.') < 1 and len(text)>0:
        sentence_text = text
        
        for exception in exceptions_list:
            sentence_text = sentence_text.replace(exception[1],exception[0])
        
        for i in range(0,10):
            for month in month_list:
                date = f"{i}%% {month}"
                replacement = f"{i}. {month}"
                sentence_text = sentence_text.replace(date,replacement)
                
        for i in range(0,len(pkt_replacements)):
            sentence_text = sentence_text.replace(pkt_replacements[i],pkt_instances[i])
                  
        sentence_list.append(sentence_text)
    return sentence_list

def get_sentence_bow_meanvector(sentence, database):
    
    tokenized = database.tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
    
    tokens = tokenized['input_ids'][0]
    
    with torch.no_grad():
        output = database.model(**tokenized, output_hidden_states=True)
        
    # extract hidden staytes        
    hidden_states = torch.cat(output[2])
    
    # mean last layer to create sentence embedding
    sentence_embedding = torch.mean(hidden_states[-1:,:,:].squeeze(), dim=0).numpy()
    
    token_embeddings = list(hidden_states[-1:,:,:].squeeze().numpy())
    
    
    words = list()
    
    for tensor, embedding in zip(tokens, token_embeddings):
        token_id = int(tensor)
        token = list(database.tokenizer.vocab.keys())[list(database.tokenizer.vocab.values()).index(token_id)]
        
        if re.match('[a-ø]',token[0]) != None:
            word = token
            
            vector = embedding 
            
            words.append([word,vector,[token_id]])
        
        elif token[0:2] == '##' and re.match('[a-ø]',token[2]) != None:
            
            if len(words) > 0:
                previous_tup = words[-1]
            else:
                print(sentence)
                print(token)
                previous_tup = ['', np.array([0]*768, dtype='float32'),[]]
            
            word = previous_tup[0] + token[2:]
            vector = (previous_tup[1] + embedding)/2
            
            words = words[:-1]
            words.append([word,vector,previous_tup[2]+[token_id]])
            
        else:
            continue
        
    bow = dict()
    for word_list in words:
        if word_list[0] in bow.keys():
            bow[word_list[0]][0] += 1
            
            bow[word_list[0]][1] = bow[word_list[0]][1]+word_list[1]
            
            bow[word_list[0]][3] += 1
            
        else:
            bow[word_list[0]] = [1, word_list[1], word_list[2], 1]
            
        
    
    
    sentence_dict = {'input_bow':bow,
                     'input_cv': sentence_embedding,
                     'full_text':sentence}
    return sentence_dict

