from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
import os
from typing import Literal

load_dotenv()

model = ChatOpenAI(
    model='alibaba/tongyi-deepresearch-30b-a3b:free',   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

parser1=StrOutputParser()

class feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Give the sentiment of the feedback')

parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template='Classify sentiment from the following text as positive ot negative \n {feedback} \n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain=prompt1|model|parser2

prompt2=PromptTemplate(
    template='Write a appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']

)

prompt3=PromptTemplate(
    template='Write a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
    
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive', prompt2|model|parser1),
    (lambda x:x.sentiment=='negative', prompt3|model|parser1),
    RunnableLambda(lambda x:"could not find sentiment")    
)

chain=classifier_chain|branch_chain

result=chain.invoke({'feedback':'This is a beautiful phone'})

print (result)
