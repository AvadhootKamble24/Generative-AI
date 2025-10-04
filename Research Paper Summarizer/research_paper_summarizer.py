from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import os
import re
os.environ['HF_HOME']=r"D:\Programming\GEN-AI\Campus x Gen AI with langchain\huggingface_cache"


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7
    )
)


model=ChatHuggingFace(llm=llm)
st.header('Research Tool📚')
paper_ip=st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_ip=st.selectbox('Select Explanation Style',["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_ip=st.selectbox('Select Explanation Length',["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

template=load_prompt('template.json')

if st.button('Sumarrize'):
    chain=template | model
    result=chain.invoke({
        'paper_input':paper_ip,
        'style_input':style_ip,
        'length_input':length_ip

    })
    # Extract only the assistant response
    full_text = result.content
    assistant_only = re.split(r"<\|assistant\|>", full_text)
    if len(assistant_only) > 1:
        # Remove any trailing special tokens like </s>
        output_text = assistant_only[1].split("</s>")[0].strip()
    else:
        output_text = full_text

    st.write(output_text)