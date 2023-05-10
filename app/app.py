import sys
sys.path.append('..')

import os
import streamlit as st
from datetime import datetime
from modules.utilities import *
import pathlib
from uuid import uuid4
import asyncio

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='Azure OpenAI Vector Search Demo', layout='wide', page_icon='../images/logo_black_simple.png')
# st.set_page_config(page_title='Azure OpenAI Cog. Search Demo', layout='wide', page_icon='../images/logo_black_simple.png')

# m = st.markdown("""
# <style>
# div.stButton > button:first-child {
#     background-color: #f0f2f6;
#     color: f0f2f6;
#     height: 30px;
#     width: 80px;
#     border-radius:5px;
#     border:1px solid f0f2f6;
#     font-size:9px;
#     font-weight: bold;
#     margin: auto;
#     display: block;
# }

# div.stButton > button:hover {
# 	border:1px solid f0f2f6;
# 	background-color:grey; 
#     color: f0f2f6;   
# }

# div.stButton > button:active {
# 	position:relative;
# 	top:3px;
# }

# </style>""", unsafe_allow_html=True)

#------------------------------------------------------------------
# #To turn off postback on focus out from text areas
# #This looks for any input box and applies the code to it to stop default behavior when focus is lost
# components.html(
#         """
#     <script>
#     const doc = window.parent.document;
#     const inputs = doc.querySelectorAll('textarea');

#     inputs.forEach(input => {
#     input.addEventListener('focusout', function(event) {
#         event.stopPropagation();
#         event.preventDefault();
#         console.log("lost focus")
#     });
#     });

#     </script>""",
#         height=0,
#         width=0,
#     )
#------------------------------------------------------------------

#To remove border from form
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
        div[data-testid="stExpander"] div[role="button"] p {font-size: 14px; font-weight:1000}
    </style>
'''
st.markdown(css, unsafe_allow_html=True)
#------------------------------------------------------------------

# Will need commenting when deploying the app
load_dotenv()

#Set env variables
setEnv()

# Exit if app set to MAINTENANCE_MODE (yes, no)
if str(os.getenv('MAINTENANCE_MODE')).lower()=='yes':    
    st.write('App is currently offline for maintenance, please check back later.')
    exit()

aoai_embedding_model = 'text-search-babbage-doc-001'#'text-search-davinci-doc-001' #'text-search-ada-doc-001'
aoai_embedding_model_version = '1'

aoai_text_model = 'text-davinci-003' 
aoai_text_model_version = '1'
aoai_text_model_temperature = 0.1
aoai_text_model_max_tokens = 300

aoai_chat_model = 'gpt-35-turbo' 
aoai_chat_model_version = '0301'
aoai_chat_model_temperature = 0.1
aoai_chat_model_max_tokens = 300
aoai_chat_model_top_p = 0.5
aoai_chat_model_max_conversations = 5
system_init_text = '''
Assistant provides a concise answer to user questions based on provided context. 

Instructions:
-Please say "not found" when answer to the question is not given the context, do not make up answers if they are not in the context. 
-Do not answer questions unrelated to the context.

Context:'''

aoai_embedding_model_deployment = aoai_deployed_models[aoai_embedding_model]["version"][aoai_embedding_model_version]["deployment_name"] #Azure OpenAI embedding deployment name
aoai_embedding_model_dim = aoai_deployed_models[aoai_embedding_model]["version"][aoai_embedding_model_version]["dim"]

aoai_text_model_deployment = aoai_deployed_models[aoai_text_model]["version"][aoai_text_model_version]["deployment_name"] #Azure OpenAI text completion deployment name

aoai_chat_model_deployment = aoai_deployed_models[aoai_chat_model]["version"][aoai_chat_model_version]["deployment_name"] #Azure OpenAI chat completion deployment name

index_name = 'vector-search-demo-index' #Cognitive Search index
api_version='2023-07-01-Preview' #Cognitive Search API
score_threshold = 60 #Show answers above or equal to this score threshold
prompt_min_length = 5
ms_alias_min_length = 6
prompt_text_area_max_chars = 300
temp_dir = '../temp_uploads/' #Where uploaded files get staged until they are indexed, files staged for few seconds only then deleted.
app_version = '0.9.8' #Equal to docker image version tag, shown in sidebar.

#--------------------------------------------------------------------------

def getKeywordList(input_text):
    input_text = input_text.replace('.',' ')
    input_text = input_text.replace('-',' ')
    input_text = input_text.replace('=',' ')
    input_text = input_text.replace('?',' ')
    input_text = input_text.replace('!',' ')
    keyword_list = [word.lower() for word in input_text.split() if word.lower() not in ['?','a','an','and','or','do','of','if','not','for','are','was','were','is','can','have','has','there','their','the','how', 'why', 'when', 'what',"what's",'in', 'to', 'i', 'we', 'you']]
    return keyword_list

def highlightKeywords(keyword_list, input_text):
    highlighted = " ".join(f'<span style="background-color: #ffff99">{t}</span>' if t.lower() in keyword_list else t for t in input_text.split(' '))    
    # print(f'highlighted:{highlighted}')   
    
    return highlighted

# print(getKeywordList('What is HREC approval timeline?'))
# print(highlightKeywords(getKeywordList("What's weather?"),'Today is a great day as we have some good weather'))

def getResult(prompt, top_n, index_name, prefix):

    out = []
    
    query_result,document_lc_list = queryCogSearchIndex(prompt=prompt, aoai_embedding_model=aoai_embedding_model_deployment, 
                                                        index_name=index_name, top_n=top_n, prefix=prefix, api_version=api_version)
    # print(f'query_result:{query_result}') 
    # print(f'document_lc_list:{document_lc_list}')
    # print(f'len(document_lc_list):{len(document_lc_list)}')

    # Check if any response received
    if document_lc_list is not None:

        # Open AI lc qna
        llm = AzureOpenAI(deployment_name=aoai_text_model_deployment,temperature=aoai_text_model_temperature, max_tokens=aoai_text_model_max_tokens)

        # lc        
        chain = load_qa_with_sources_chain(llm, chain_type="map_rerank", verbose=False, return_intermediate_steps=True)
        chain_out = chain({"input_documents": document_lc_list, "question": prompt}, return_only_outputs=False)
        # print(f'chain_out:{chain_out}')
        print(f'len(chain_out["input_documents"]):{len(chain_out["input_documents"])}')

        results = []
        for i, item in enumerate(chain_out['intermediate_steps']):
            # print(item['answer'], item['score']) #Uncomment to view the answer
            # results.append((int(item['score']),i,item['answer']))
            results.append((int(item['score']),i,item['answer'],chain_out['input_documents'][i].metadata['@search.score']))
            
            # print(f'item {str(i)}:{item}')
            # print(f'chain_out["input_documents"] {str(i)}:{chain_out["input_documents"][i]}')            
            # print(f"@search.score:{chain_out['input_documents'][i].metadata['@search.score']}")

        # results.sort(reverse = True) #Sort desc based on Score
        results = sorted(results, key = lambda x: (x[0], x[3]), reverse=True) #Sort desc based on-> score and @search.score
        # print(f'results:{results}')
        # # print(results[0][1]) #top first answer index    

        # Top N answers
        # for i in range(top_n):
        for i in range(len(chain_out["input_documents"])):
            
            # # Uncomment to debug
            # print(f'i:{str(i)}')
            # print(f"\nAnswer: {results[i][2]}") #answer 
            # print(f"Score: {results[i][0]}") #answer score
            # print(bcolors.OKGREEN+f"Content {str(i+1)}: {chain_out['input_documents'][results[i][1]].page_content}"+bcolors.ENDC) #content
            # print(bcolors.BOLD+f"Source {str(i+1)}: {chain_out['input_documents'][results[i][1]].metadata['source']}"+bcolors.ENDC) #source
            # print(f"@search.score {str(i+1)}: {chain_out['input_documents'][results[i][1]].metadata['@search.score']}") #similarity score
            # print(f"Page: {int(chain_out['input_documents'][results[i][1]].metadata['page'])+1}") #page

            # Check score threshold
            if int(results[i][0]) >= score_threshold:
                out_item = None
                out_item = {
                    "Answer":results[i][2],
                    "Score": int(results[i][0]),
                    f"Content": chain_out['input_documents'][results[i][1]].page_content,
                    f"Source": chain_out['input_documents'][results[i][1]].metadata['source'],
                    f"@search.score": chain_out['input_documents'][results[i][1]].metadata['@search.score'],
                    f"Page": int(chain_out['input_documents'][results[i][1]].metadata['page'])+1
                    }
                out.append(out_item)        
    

    return out    

#--------------------------------------------------------------------------

# Initialization of session vars
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'answers' not in st.session_state:
    st.session_state['answers'] = []
if 'search_click_answer' not in st.session_state:
    st.session_state['search_click_answer'] = []
if 'user_chat_dict' not in st.session_state:
    st.session_state['user_chat_dict'] = {"j":{}}
if 'chat_processing_completed' not in st.session_state:
    st.session_state['chat_processing_completed'] = True #Add to prevent processing if Submit clicked while processing chat completion request


with st.container():
    
    def upload_button_click():

        if file_uploader is not None and len(textbox_msalias.strip()) >= ms_alias_min_length:
            progress_bar = middle_column_12.progress(0,'')
            
            # st.write(str(os.listdir('../')))
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # print(file_uploader.getvalue())
            # local_file = pathlib.Path('./temp_uploads/'+str(uuid4())+'_'+file_uploader.name)
            local_file = pathlib.Path(temp_dir + file_uploader.name)            
            local_file.write_bytes(file_uploader.getvalue()) #Write locally to crack open PDF/word docs
            
            local_file_path = str(local_file)
            # print(local_file_path)

            progress_bar.progress(20,'File acquired')

            # # Crack open PDF doc
            # document_pages_lc = readPDF(local_file_path)
            # print((document_pages_lc[0]))            

            progress_bar.progress(30,'Processing')
            
            # Read document, cleanse content, get content and embeddings
            document_page_content_list, \
            document_page_embedding_list, \
            document_page_no_list = getEmbeddingEntireDoc(documentPath=local_file_path, 
                                                        aoai_embedding_model=aoai_embedding_model_deployment, 
                                                        chunk_size=1)
            print('Embeddings retrieved')
            print(len(document_page_content_list), len(document_page_embedding_list), len(document_page_no_list))
            # print(document_page_content_list)
            # print(document_page_embedding_list, document_page_no_list)

            progress_bar.progress(80,'Almost done')

            # Add document pages            
            response = addDocumentToCogSearchIndex(index_name=index_name,documentPath=local_file_path, 
                                                       document_page_content_list=document_page_content_list, 
                                           document_page_embedding_list=document_page_embedding_list, 
                                           document_page_no_list=document_page_no_list, 
                                           prefix=textbox_msalias,
                                           api_version=api_version)

            print(f'addDocumentToCogSearchIndex: {response}')

            progress_bar.progress(90,'Running cleanup')

            # Remove local PDF after indexing completed
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

            progress_bar.progress(100,'Completed')

        if len(textbox_msalias.strip()) < ms_alias_min_length:
            left_column.warning('Please enter a valid alias')

    top_left_column, middle_left_column, right_left_column = st.columns([40,20,40])
    top_left_column_1, top_left_column_2 = top_left_column.columns([25,75])
    top_left_column_1.image(image='../images/logo_black.png', width=100)
    # top_left_column_2.write('###')
    
    top_left_column_2.subheader('Cognitive Vector Search Demo')    
    # top_left_column_2.subheader('Cognitive Search Demo')

    top_left_column_2.write('Unleash the power of your documents with data-driven inquiries')    

    # st.write('---')   

    with st.sidebar:      
                    
        st.markdown(':gear: Settings')

        textbox_msalias = st.text_input(label='Unique alias*', max_chars=10, key='textbox_msalias', type='password', 
                                        help='''Unique text value to store/query your docs under. 
                                        Use same value when you revisit this app in future for consistent experience.''')
        selectbox_top_n = st.selectbox(label='Top N results',options=(3,5,10), index = 1, key='selectbox_top_n')        

        checkbox_score = st.checkbox(label='Answer Score',key='checkbox_score', value=False, help='Value between 0 to 100 suggesting LLM confidence for answering the question by with retrieved passage of text.')
        checkbox_similarity = st.checkbox(label='Search Score',key='checkbox_similarity', value=False, help='Content search score.')   

        checkbox_page_no = st.checkbox(label='Page No',key='checkbox_page_no', value=True, help='Document page number.')    
        checkbox_show_fileupload = st.checkbox(label='Upload file',key='checkbox_show_fileupload', value=False, help='Upload file using upload widget.')

        st.write('---')
        st.write('Q&A Assistant')
        selectbox_max_conversations = st.selectbox(label='Max conversations',options=(2,5), index = 1, key='selectbox_max_conversations')
        selectbox_max_tokens = st.selectbox(label='Max tokens',options=(100,300,500), index = 2, key='selectbox_max_tokens')

        st.write('### \n ### \n ### \n ### \n ###')
        st.write("[Github Repo - TBA]()")
        st.caption('Version: '+app_version)
        st.write('<p style="font-size:14px; color:black;"><b>Powered by Azure OpenAI</b></p>', unsafe_allow_html=True)
    
    if checkbox_show_fileupload == True:
        st.write('----')

        left_column_11, middle_column_12, right_column_13 = st.columns([36,8,56])
        file_uploader = left_column_11.file_uploader(label='Upload file.',accept_multiple_files=False, key='file_uploader_1',type=['pdf', 'docx'],label_visibility='hidden')    
        middle_column_12.write('###')
        middle_column_12.write('###')
        # middle_column_b.write('###')    
        upload_button = middle_column_12.button(label='Upload', on_click=upload_button_click, key='btn_upload')       
        
        #fffce7
        right_column_13.write('''<b><u>Disclaimer</u></b> 
                        \n <p style="font-size:16px; color:black;background-color:#f7b0b0">This public demo app is <b>not intended for use with sensitive data</b>. 
                        We strongly advise against uploading any sensitive data to this application. 
                        We cannot guarantee the security of any data uploaded to this application. By using this application, you acknowledge that you understand and accept this risk.
                        Please use <b>publicly available data</b> only.</p>
                        \n <p style="font-size:16px; color:black;background-color:#f7b0b0"><i>For use with sensitive documents, please clone the repository and run it in your own environment.</i></p>'''
                        ,unsafe_allow_html=True)
        st.write('----')       
        

with st.container():

    # left_column, middle_column, right_column = st.columns([46,8,46])
    # left_column, middle_column, right_column = st.columns([60,10,30])
    left_column, middle_column, right_column = st.columns([90,10,1])
    
    prompt = left_column.text_area(label='Enter your question:',max_chars=prompt_text_area_max_chars, key='text_area1', label_visibility ='hidden')       
   
    questions = st.session_state['questions']
    answers = st.session_state['answers']
    answer = st.session_state['search_click_answer']
    
    def search_click():
        if len(textbox_msalias.strip()) < ms_alias_min_length:
            left_column.warning('Please enter a valid alias')

        elif prompt is not None and len(prompt.strip()) >= prompt_min_length and len(textbox_msalias.strip()) >= ms_alias_min_length:
            answer = []
            
            top_n = int(selectbox_top_n)

            try:
                answer = getResult(prompt, top_n, index_name, textbox_msalias)
                st.session_state['search_click_answer'] = answer
            except Exception as e:
                print(f'Error getResult(): {e}')
                print('Exception in getResult()')

            #No results retrieved
            if len(answer)==0:
                left_column.warning('No results found. Consider uploading document/s first if you are using this app for the first time for unique alias you have specified. \n Check Upload file --> Browse file --> Click Upload to get started.')

    
    #Populate bottom pane with all N responses (Either new answer or answer from session state)
    for j, ans_details in enumerate(answer):

        keyword_list = getKeywordList(prompt)                
        
        left_column.write(f'<div style="font-size:16px; color:black;background-color:#93bfe6;border:1px solid #93bfe6;border-top-left-radius:5px;border-top-right-radius:5px;padding:5px"> \
                          <b>Answer</b>: \
                            {ans_details["Answer"]} </div>',
                            unsafe_allow_html=True)             

        if checkbox_score:
            left_column.write(f'<p style="font-size:12px; color:black"><b>Score</b>: {ans_details["Score"]}</p>',unsafe_allow_html=True)

        # left_column.write(f'<p style="font-size:16px; color:black;background-color:#e8ebfa""><b>Content</b>: {ans_details[f"Content"]}</p>',unsafe_allow_html=True)
        # left_column.write(f'<div style="font-size:16px; color:black;background-color:#e8ebfa""><b>Content</b>:<br> {highlightKeywords(keyword_list, ans_details[f"Content"])}</div>',unsafe_allow_html=True)                        
        # left_column.markdown(f'<b>Content</b>:<br> {highlightKeywords(keyword_list, ans_details[f"Content"])}',unsafe_allow_html=True)                
        content_text = f'<b>Content</b>:<br> {highlightKeywords(keyword_list, ans_details[f"Content"])}'
        left_column.write(f'<div style="font-size:16px; color:black;background-color:#e6eaed;border:1px solid #93bfe6;border-bottom-left-radius:5px;border-bottom-right-radius:5px;padding-left:5px;padding-right:5px">{content_text}</div>',unsafe_allow_html=True) 
        
        left_column.write(f'<div><br></div>',unsafe_allow_html=True)
        
        # left_column_a, left_column_b =  left_column.columns([85,15])
        left_column.write(f'<p style="font-size:14px; color:black"><b>Source</b>:<i> {os.path.basename(ans_details[f"Source"])}</i></p>',unsafe_allow_html=True)
        # analyse_button = left_column_b.button(label='Analyse', key=f'btn_analyse_{str(j)}')    
                        
        if checkbox_similarity:
            left_column.write(f'<p style="font-size:12px; color:black"><b>@search.score</b>: {ans_details[f"@search.score"]}</p>',unsafe_allow_html=True)
        
        if checkbox_page_no:
            left_column.write(f'<p style="font-size:12px; color:black"><b>Page No</b>: {ans_details[f"Page"]}</p>',unsafe_allow_html=True)                
        
        # ---- Q&A Assistant  ----
        
        #Do not process if user submits while previous request is already in-progress 
        if st.session_state['chat_processing_completed']:
                         
            expander = left_column.expander("Analyse with Q&A Assistant", expanded=False)

            with expander:                

                with st.form(key= f'chat_form_{str(j)}',clear_on_submit=True):                   
                    
                    chat_log_control = st.empty()
                    chat_gpt_query = st.text_area(label='User message:',max_chars=prompt_text_area_max_chars, key=f'text_area_2_{str(j)}', label_visibility ='hidden', height=20)    
                        
                    content = ans_details[f"Content"]                   

                    chat_column_a_btn1,chat_column_a_btn2, chat_column_a_btn3 = st.columns([8,8,84])
                    submitted = chat_column_a_btn1.form_submit_button(label='Submit')
                    reset = chat_column_a_btn2.form_submit_button(label='Reset')
                    
                    if submitted: 
                        st.session_state['chat_processing_completed'] = False
                        # print('In submit_click()')
                        # print(f'chat_gpt_query:{chat_gpt_query}')

                        with st.spinner(text = "Analysing.."):

                            user_input_list = []
                            assistant_output_list = []     

                            if chat_gpt_query is not None and len(chat_gpt_query) > 0:

                                chat_log_text = ''

                                #Check if key exists / was set in dictionary
                                if j in st.session_state['user_chat_dict']["j"]:
                                    user_input_list = st.session_state['user_chat_dict']["j"][j]["user_input_list"]
                                    assistant_output_list = st.session_state['user_chat_dict']["j"][j]["assistant_output_list"]

                                user_input_list.append(chat_gpt_query)

                                aoai_chat_model_max_tokens = int(selectbox_max_tokens)

                                assistant_output = getChatCompletion(system_init_text = system_init_text + content,
                                                user_input_list=user_input_list,
                                                assistant_output_list=assistant_output_list, 
                                                aoai_chat_model=aoai_chat_model_deployment, 
                                                aoai_chat_model_temperature=aoai_chat_model_temperature, 
                                                aoai_chat_model_max_tokens=aoai_chat_model_max_tokens, 
                                                aoai_chat_model_top_p=aoai_chat_model_top_p)
                                
                                assistant_output_list.append(assistant_output)

                                # print(f'user_input_list:{user_input_list}')
                                # print(f'assistant_output:{assistant_output_list}')

                                for i, _ in enumerate(user_input_list):

                                    chat_log_text = chat_log_text + \
                                    '<table style="border:hidden;padding: 5px;width:100%;">' + \
                                    '<tr><td style="border:hidden;"></td><td align="right" style="border:hidden;width:50%;">' +\
                                    f'<div style="font-size:16px; color:black;background-color:#e6eaed;border:1px solid #e6eaed;border-radius: 10px;padding: 5px; width:95%;" align="right">{str(user_input_list[i])}</div>' + \
                                    '</td></tr>'+\
                                    '<tr><td  align="left" style="border:hidden;width:50%;">' + \
                                    f'<div style="font-size:16px; color:black;background-color:#93bfe6; width:50%;border:1px solid #93bfe6;border-radius: 10px;padding: 5px; width:95%;" align="left">{str(assistant_output_list[i])}</div>' +\
                                    '</td><td style="border:hidden;"></td></tr></table>'                                   
                                    
                                chat_log_control.write(chat_log_text, unsafe_allow_html=True)

                                #Only keep last N conversation history                                
                                aoai_chat_model_max_conversations = int(selectbox_max_conversations)-1

                                if len(user_input_list) > aoai_chat_model_max_conversations:
                                    user_input_list = user_input_list[-aoai_chat_model_max_conversations:]
                                    print('user_input_list trimmed.')
                                if len(assistant_output_list) > aoai_chat_model_max_conversations:
                                    assistant_output_list = assistant_output_list[-aoai_chat_model_max_conversations:]
                                    print('assistant_output_list trimmed.')
                                
                                #Set the list in session
                                st.session_state['user_chat_dict'] = {
                                                    "j": 
                                                        { j:
                                                            {"user_input_list":user_input_list, 
                                                            "assistant_output_list":assistant_output_list
                                                            }
                                                        }
                                                    }                           
                             
                            
                        # Mark chat completion processing complete
                        st.session_state['chat_processing_completed'] = True

                    if reset:
                        print('In reset')
                        st.session_state['user_chat_dict'] = {"j":{}}                        
                        st.session_state['chat_processing_completed'] = True
                        chat_log_control.empty()

            
        # ---- Q&A Assistant  ----  
                                         
        else:
            left_column.warning('Processing interrupted, please click Clear button to reset the session.')
            
        # left_column.write('----')

    if str(prompt).strip() != '' and len(answer) > 0:
        questions.append(prompt)
        answers.append(answer)        

    st.session_state['questions'] = questions          
    st.session_state['answers'] = answers


        # print(f'questions:{questions}')
        # print(f'answers:{answers}')

        # #Commenting below, as this space will be used for showing the Chat interface.
        # if len(list(reversed(questions))) > 0:
        #     right_column.write(f'<p style="font-size:16px; color:black"><b>Question History</b></p>',unsafe_allow_html=True)    
            

        # #Commenting below, as this space will be used for showing the Chat interface.
        # # Show in reversed order without modifying the lists set into sessions
        # for i, item in enumerate(list(reversed(questions))):                       

        #     question_text = str('' + str(list(reversed(questions))[i]))
        #     # answer_text = str(''+ str(list(reversed(answers))[i]))
            
        #     # [{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}]
        #     # right_column.write('###')
            
        #     right_column.write(f'<p style="font-size:16px; color:black;background-color:#f0f2f6"><b>Question</b>: {question_text} </p>',unsafe_allow_html=True)
        #     # right_column.write(f'<p style="font-size:14px; color:black"><b>Answer</b>: {answer_text}</p>',unsafe_allow_html=True)
        #     # right_column.write('---')
        #     # print(list(reversed(answers))[i])
        #     for j, ans_details in enumerate(list(reversed(answers))[i]):                
                
        #         #Only show 1 top answer in history (right side pane)
        #         if j==0:
        #             right_column.write(f'<p style="font-size:14px; color:black"><b>Answer</b>: {ans_details["Answer"]}</p>',unsafe_allow_html=True)
        #             if checkbox_score:
        #                 right_column.write(f'<p style="font-size:12px; color:black"><b>Score</b>: {ans_details["Score"]}</p>',unsafe_allow_html=True)
        #             right_column.write(f'<p style="font-size:12px; color:black"><b>Content</b>: {ans_details[f"Content"]}</p>',unsafe_allow_html=True)
        #             right_column.write(f'<p style="font-size:12px; color:black"><b>Source</b>:<i> {os.path.basename(ans_details[f"Source"])}</i></p>',unsafe_allow_html=True)
        #             if checkbox_similarity:
        #                 right_column.write(f'<p style="font-size:12px; color:black"><b>@search.score</b>: {ans_details[f"@search.score"]}</p>',unsafe_allow_html=True)
        #             if checkbox_page_no:
        #                 right_column.write(f'<p style="font-size:12px; color:black"><b>Page No</b>: {ans_details[f"Page"]}</p>',unsafe_allow_html=True)                
        #             right_column.write('---')               


    def clear_click():
        st.session_state['text_area1'] = ''   
        st.session_state['questions'] = []
        st.session_state['answers'] = []
        # st.session_state['checkbox_score'] = False
        # st.session_state['checkbox_similarity'] = False
        # st.session_state['checkbox_page_no'] = False
        st.session_state['search_click_answer'] = []
        st.session_state['user_chat_dict'] = {"j":{}}
        st.session_state['chat_processing_completed'] = True
    
       
    middle_column.write('###')
    middle_column.write('###')
    search_button= middle_column.button(label='Search', on_click= search_click, key='btn_search')
    clear_button = middle_column.button(label='Clear', on_click = clear_click, key='btn_clear')    