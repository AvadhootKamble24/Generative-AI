from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

model1 = ChatOpenAI(
    model='alibaba/tongyi-deepresearch-30b-a3b:free',   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

model2= ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL"),      #nvidia/nemotron-nano-9b-v2   
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",  # Important: points to OpenRouter
    temperature=0.7,
    # max_tokens=400,
)

prompt1=PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate 5 quiz question from the following text \n {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge  both notes and quiz question in a single document. notes-> {notes} and quiz->{quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})

merge_chain=prompt3|model2|parser

chain=parallel_chain|merge_chain

text="""
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
""" 

result= chain.invoke({'text':text})

print(result)