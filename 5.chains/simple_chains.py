from langchain_openai import  ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os 
load_dotenv()
prompt=PromptTemplate(
    template='generate 5 intresting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)
parser=StrOutputParser()

chain=prompt|model|parser

result=chain.invoke({'topic':'cricket'})
print(result)

