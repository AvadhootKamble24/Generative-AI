from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

os.environ['HF_HOME']=r"D:\Programming\GEN-AI\Campus x Gen AI with langchain\huggingface_cache"


llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.6,
        max_new_tokens=100
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke('What is capital of india? ')

print(result.content)