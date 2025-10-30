from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
url="https://www.amazon.in/Apple-iPhone-15-128-GB/dp/B0CHX2F5QT?pd_rd_w=qHJtS&content-id=amzn1.sym.a324903e-1f30-4243-bf0d-6da5ebc52115&pf_rd_p=a324903e-1f30-4243-bf0d-6da5ebc52115&pf_rd_r=RZQN1W9Y08TCHRD1ZT6Y&pd_rd_wg=XbqJZ&pd_rd_r=e35921af-b780-4247-90ed-9aa032b79565&pd_rd_i=B0CHX2F5QT&ref_=pd_hp_d_btf_unk_B0CHX2F5QT&th=1"

loader= WebBaseLoader(url)

parser=StrOutputParser()

docs=loader.load()

model= ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),      #nvidia/nemotron-nano-9b-v2   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

prompt=PromptTemplate(
    template='Answer the following question \n {question} fromthe following text -\n {text}'
    input_variables=['question','text']
)

print(len(docs))

chain=prompt|model|parser
result=chain.invoke({'question':'what is the product the we are talking about?  ','text':docs[0].page_content})