from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader=DirectoryLoader(
    path='pdfs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.load() # loads all the documents in memory at once
docs=loader.lazy_load() # loads the documents as per need in memory at once

for document in docs:
    print(document.metadata)
# print(docs[0].page_content)
# print(docs[3].metadata)

