from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2= PromptTemplate(
    template='write 5 line summary on the following text. /n {text}',
    input_variables=['text']
    
)

parser=StrOutputParser()

chain=template1 | model |template2| model | parser

result= chain.invoke({"topic":"black hole"})

print(result)
