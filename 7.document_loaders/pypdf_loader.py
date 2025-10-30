from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader(r'D:\Programming\GEN-AI\Campus x Gen AI with langchain\By_Nitish_Singh_github\langchain-document-loaders\dl-curriculum.pdf')

docs=loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)