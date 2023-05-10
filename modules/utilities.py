#------------Imports---------------
import os

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.llms.openai import AzureOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

import openai
from dotenv import load_dotenv

import hashlib
import numpy as np

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

import requests
import json
import logging

logging.basicConfig(level=logging.ERROR)

#------------Functions---------------

'''
Initialise environment variables
'''
def setEnv():
    try:
        openai.api_type = os.getenv('OPENAI_API_TYPE')
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_version = os.getenv('API_VERSION')
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        return True
    except Exception as e:
        logging.error(f'Error setEnv(): {e}')    
        return False

'''
documentPath: Path to document (pdf/word/etc.)
'''
def getDocumentExtension(documentPath):
    try:
        return os.path.basename(documentPath).split('.')[len(os.path.basename(documentPath).split('.'))-1]
    except Exception as e:
        logging.error(f'Error getDocumentExtension(): {e}')    
        return None

'''Read PDF documents and return the list of langchain documents
'''
def readPDF(source_url):
    try:
        document_pages_lc = None
        document_pages_lc = PyPDFLoader(source_url).load()

        # for page in document_pages_lc:
            
        #     print(f'Source: {str(page.metadata["source"])}')
        #     print(f'Page: {str(int(page.metadata["page"])+1)}')
        #     print(page.page_content)

        return document_pages_lc
    except Exception as e:
        logging.error(f'Error readPDF(): {e}')
        return None

'''Read MS Word documents and return the list of langchain documents
'''
def readMSWord(source_url):
    try:
        one_page_size = 300 #IMP: How many words per split page of whole doc.
        document_pages_lc = None
        document_pages_lc = UnstructuredWordDocumentLoader(source_url).load() #Note: This method does not return same object as PDf loader, e.g. Doc pages not recognized. So below custom logic is built.
        document_pages_lc_list = []        
        
        # UnstructuredWordDocumentLoader returns whole doc as a single page, so need to impelement custom splitting
        for page in document_pages_lc:                       
            
            page_words = page.page_content.split(' ') #Split doc into words

            #Split document into pages of one_page_size words each
            for i in range((len(page_words) // one_page_size)+1):
                # print(i)
                
                # Note: Replaced below with Document object as in code below this section.
                # document_pages_lc_dict = {} #{"page_content":"",metadata={"source": "..doc", "page": 4}}
                # document_pages_lc_dict["page_content"] =  ' '.join(page_words[i*one_page_size:(i+1)*one_page_size])
                # document_pages_lc_dict["metadata"] = {"source":page.metadata["source"], "page":i}
                # document_pages_lc_list.append(document_pages_lc_dict)     

                doc = Document(page_content=' '.join(page_words[i*one_page_size:(i+1)*one_page_size]),
                               metadata={"source":page.metadata["source"], "page":i})
                document_pages_lc_list.append(doc)                   
        
        return document_pages_lc_list
    except Exception as e:
        logging.error(f'Error readMSWord_old(): {e}')
        return None
    
'''
Removes new line characters, double spaces
input_text: Piece of text
'''
def cleanseText(input_text):
    try:
        input_text_cleansed = None
        input_text_cleansed = input_text.replace('\n',' ') #Remove new line characters
        input_text_cleansed = input_text_cleansed.replace('  ',' ') #Remove double space

        return input_text_cleansed
    except Exception as e:
        logging.error(f'Error cleanseText(): {e}')
        return None

'''
txt_data: input data
aoai_embedding_model: Azure OpenAI deployment name
chunk_size: Maximum number of texts to embed in each batch
max_retries: Maximum number of retries to make when generating.
'''
def getEmbedding(txt_data, aoai_embedding_model, chunk_size=1, max_retries = 2):
    try:
        embeddings = OpenAIEmbeddings(model=aoai_embedding_model, chunk_size=chunk_size, max_retries=max_retries)        
        query_result = embeddings.embed_query(txt_data)
        return query_result
    except Exception as e:
        logging.info(f'txt_data: {txt_data}')
        logging.error(f'Error getEmbedding(): {e}')        
        return None
    
'''
Generate embedding for entire doc
documentPath: Path to the document
'''
def getEmbeddingEntireDoc(documentPath, aoai_embedding_model, chunk_size=1):

    try:
        docType = None
        document_pages_lc = None
        document_page_embedding_list = []    
        document_page_content_list = []
        document_page_no_list = []

        #Get document type
        docType = getDocumentExtension(documentPath).lower()
        
        if docType == 'pdf':
            document_pages_lc = readPDF(documentPath)

        # Custom word doc processing as there's not page metadata like PDF loader, 
        # also the doc is not split into pages like PDF does out of the box. Please review readMSWord() method for more details.
        elif docType == 'docx' or docType == 'doc':
            document_pages_lc = readMSWord(documentPath)
        
        for document_page in document_pages_lc:
            # print(document_page)
            # print(document_page.page_content)
            # print(document_page.metadata["source"])
            # print(document_page.metadata["page"])

            source_doc_path = None
            source_doc_page_no = None
            source_doc_page_content = None
            embedding_result = None

            # if docType == 'pdf':
            #     source_doc_path = document_page.metadata["source"]
            #     source_doc_page_no = int(document_page.metadata["page"])
            #     source_doc_page_content = document_page.page_content

            # elif docType == 'docx' or docType == 'doc':
            #     source_doc_path = document_page["metadata"]["source"]
            #     source_doc_page_no = int(document_page["metadata"]["page"])
            #     source_doc_page_content = document_page["page_content"]

            source_doc_path = document_page.metadata["source"]
            source_doc_page_no = int(document_page.metadata["page"])
            source_doc_page_content = document_page.page_content
            
            # print(source_doc_path)
            # print(source_doc_page_no)
            # print(source_doc_page_content)

            source_doc_page_content_cleansed = cleanseText(source_doc_page_content)

            if (source_doc_page_content_cleansed) is not None and (len(source_doc_page_content_cleansed)>0) and (source_doc_page_content_cleansed.strip != ''):                    

                embedding_result = getEmbedding(source_doc_page_content_cleansed, aoai_embedding_model, chunk_size=1, max_retries = 3)
                # print(embedding_result)

                if embedding_result is not None:
                    document_page_content_list.append(source_doc_page_content) #Retain formatting
                    document_page_embedding_list.append(embedding_result)        
                    document_page_no_list.append(source_doc_page_no)
                else:
                    print(f'Unable to embed text:{source_doc_page_content}, moving to next.')
                    logging.warning(f'Unable to embed text:{source_doc_page_content}, moving to next.')

        return document_page_content_list, document_page_embedding_list, document_page_no_list
    except Exception as e:
        logging.error(f'Error getEmbeddingEntireDoc(): {e}')
        return None, None, None   

'''
Add record to Cognitive Search index
'''
def addCogSearchIndexRecord(index_name, id, page_content, page_content_vector, page_number, documentPath, prefix = 'doc',api_version='2023-07-01-Preview'):
    try:    

        endpoint = os.getenv('COG_SEARCH_ENDPOINT')
        key = os.getenv('COG_SEARCH_ADMIN_KEY')

        DOCUMENT = {
            "id":str(id),
            "prefix":str(prefix),
            "document_path":str(documentPath),
            "page_content": str(page_content),
            "page_number":int(page_number),
            "page_content_vector": page_content_vector
        }

        search_client = SearchClient(endpoint=endpoint, index_name=index_name,credential=AzureKeyCredential(key),api_version=api_version)
        result = search_client.upload_documents(documents=[DOCUMENT])
        # logging.info("Upload of new document succeeded: {}".format(result[0].succeeded))       
        
        return True

    except Exception as e:        
        logging.error(f'Error addCogSearchIndexRecord(): {e}')
        return False

'''
Iterate over read document and add it to the index
'''
def addDocumentToCogSearchIndex(index_name, documentPath, document_page_content_list, document_page_embedding_list, document_page_no_list, prefix,api_version):
    try:
        
        # Iterate through pages
        for i, embedding in enumerate(document_page_embedding_list):   

            hash_key = hashlib.sha1(f'{documentPath}_{i}'.encode('utf-8')).hexdigest()         
            
            addCogSearchIndexRecord(index_name=index_name,
                                id = hash_key,                                 
                                page_content = document_page_content_list[i], 
                                page_content_vector = document_page_embedding_list[i], 
                                page_number = document_page_no_list[i], 
                                prefix = prefix,
                                documentPath = documentPath,
                                api_version=api_version
                                )    
                         

        return True
    except Exception as e:
        logging.error(f'Error addDocumentToCogSearchIndex(): {e}')
        return False
    
'''
Run search query.
'''
def queryCogSearchIndex(prompt, aoai_embedding_model, index_name, top_n, prefix, api_version):
    try:

        document_lc_list = []        
        query_result = None                     

        vec_prompt = getEmbedding(txt_data=prompt, aoai_embedding_model=aoai_embedding_model, chunk_size=1, max_retries = 3)
        # print('vec_prompt generated.')      

        endpoint = os.getenv('COG_SEARCH_ENDPOINT')
        search_query_key = os.getenv('COG_SEARCH_ADMIN_KEY')
        
        # #Python implementation does not support search using vector yet, so REST implementation below.
        # client = SearchClient(endpoint=endpoint, index_name=index_name,credential=AzureKeyCredential(key),api_version=api_version)
        # query_result = client.search(search_text = 'MLS',
        #                              select=['page_content','page_number', 'document_path'], 
        #                              filter = f"prefix eq '{prefix}'",
        #                              top=top_n)        
        
        api_url = endpoint+'/indexes/'+index_name+'/docs/search?api-version='+api_version
        # print(f'api_url:{api_url}')

        headers = {
            "Content-Type":"application/json",
            "api-key":search_query_key
        }

        body = {
            "vector": 
            {
                "value": vec_prompt,
                "fields": "page_content_vector",
                "k": top_n
            },
            "select": "page_content,page_number,document_path",
            "filter":f"prefix eq '{prefix}'",
            "top": top_n,
            "orderby": 'search.score() desc'
        }

        query_result = requests.post(api_url, data=json.dumps(body), headers=headers)
        print(f'query_result.status_code:{query_result.status_code}')
        query_result=query_result.json()        

        #Create lc document, for use with lc        
        for value in query_result['value']:
            document_lc = Document(page_content=str(value["page_content"]),
                                   metadata={"source":str(value["document_path"]), 
                                             "page": int(value["page_number"]), 
                                             "@search.score":float(value["@search.score"])})
            document_lc_list.append(document_lc)

        return query_result, document_lc_list


    except Exception as e:
        logging.error(f'Error queryCogSearchIndex(): {e}')
        return None, None
    
'''
Get Chat completion.
'''
def getChatCompletion(system_init_text,user_input_list,assistant_output_list, aoai_chat_model, aoai_chat_model_temperature, aoai_chat_model_max_tokens, aoai_chat_model_top_p):
    try:
        #Create ChatML
        prompt = ''
        prompt += '<|im_start|>system\n' + system_init_text + '\n<|im_end|>\n'

        for i, item in enumerate(user_input_list):
            if i < (len(user_input_list) - 1):
                 prompt += '<|im_start|>user\n' + user_input_list[i] + '\n<|im_end|>\n'
                 prompt += '<|im_start|>assistant\n' + assistant_output_list[i] + '\n<|im_end|>\n'
            else:
                prompt += '<|im_start|>user\n' + user_input_list[i] + '\n<|im_end|>\n'
                prompt += '<|im_start|>assistant\n' 

        # print(f'IM prompt:{prompt}')

        response = openai.Completion.create(
            engine=aoai_chat_model, # The deployment name you chose when you deployed the ChatGPT model
            prompt=prompt,
            temperature=aoai_chat_model_temperature,
            max_tokens=aoai_chat_model_max_tokens,
            top_p=aoai_chat_model_top_p,
            stop=["<|im_end|>"])
        
        query_result = response['choices'][0]['text']        
        # print(f'response:{response}')
        return query_result
    
    except Exception as e:        
        logging.error(f'Error getChatCompletion(): {e}')        
        return None
    
#-----------------------------------
# Functions end here.
#-----------------------------------

#For cmd background colour
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#-----------------------------------

aoai_deployed_models = {

    "text-search-ada-doc-001":{
        "version":{
            "1":{
                "deployment_name": "text-search-ada-doc-001-v1",
                "dim": 1024    
                }
            }    
        },

    "text-search-babbage-doc-001":{
        "version":{
            "1":{
                "deployment_name": "text-search-babbage-doc-001-v1",
                "dim": 2048    
                }
            }    
        },

    "text-search-curie-doc-001":{
        "version":{
            "1":{
                "deployment_name": "text-search-curie-doc-001-v1",
                "dim": 4096    
                }
            }    
        },

    "text-search-davinci-doc-001":{
        "version":{
            "1":{
                "deployment_name": "text-search-davinci-doc-001-v1",
                "dim": 12288    
                }
            }    
        },

    "text-embedding-ada-002":{
        "version":{
            "1":{
                "deployment_name": "text-embedding-ada-002-v1",
                "dim": 1536    
                }
            }    
        },

    "text-davinci-003":{
        "version":{
            "1":{
                "deployment_name": "text-davinci-003-v1"                
                }
            }    
        },

    "gpt-35-turbo":{
        "version":{
            "0301":{
                "deployment_name": "gpt-35-turbo-v0301"                
                }
            }    
        }

    }