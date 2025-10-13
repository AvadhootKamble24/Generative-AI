from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

doc =['Delhi is capital of India',
      'Kolkatta is capital of India',
      'Paris is capital of france']

vector=embedding.embed_documents(doc)

print(str(vector))