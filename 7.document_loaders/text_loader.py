from langchain_community.document_loaders import TextLoader #u can find all the doc loaders in community package
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model= ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),      #nvidia/nemotron-nano-9b-v2   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

prompt=PromptTemplate(
    template='Write a summary for the following poem \n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

loader=TextLoader(r'D:\Programming\GEN-AI\Campus x Gen AI with langchain\By_Nitish_Singh_github\langchain-document-loaders\cricket.txt',encoding='utf-8')

docs=loader.load() #loads file as a document

'''
print(type(docs)) #list

print(len(docs)) # 1

print(type(docs[0])) # langchain_core.documents.base.Document 

print(docs[0].page_content) #get only page content
'''

chain=prompt|model|parser

result=chain.invoke({'poem':docs[0].page_content})

print(result)