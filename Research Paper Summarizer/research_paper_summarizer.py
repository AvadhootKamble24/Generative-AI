import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt

#Load environment variables
load_dotenv()

#Initialize model
model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

#Streamlit UI
st.header("📚 Research Paper Summarizer (Grok-powered)")

paper_ip = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ]
)

style_ip = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_ip = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

#Load prompt
template = load_prompt(r"D:\Programming\GitHub\Generative AI\Research Paper Summarizer\template.json")

#On button click
if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_ip,
        "style_input": style_ip,
        "length_input": length_ip,
    })

    #Extract assistant-only response if model outputs user/assistant tags
    full_text = result.content
    assistant_only = re.split(r"<\|assistant\|>", full_text)
    if len(assistant_only) > 1:
        output_text = assistant_only[1].split("</s>")[0].strip()
    else:
        output_text = full_text.strip()

    st.markdown("### 🧠 Summary:")
    st.write(output_text)
