# qa_engine.py
# Perform Retrieval-Augmented Generation using LangChain and Mistral LLM

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from vector_store import load_vector_store


def build_qa_chain():
    """
    Builds a RetrievalQA chain with custom prompt and Mistral LLM.
    :return: LangChain RetrievalQA object
    """
    vectordb = load_vector_store()

    # Define custom prompt template
    template = """
    Use the following pieces of context to answer the user's question as accurately as possible.
    If you don't know the answer, just say you don't know, don't try to make up an answer.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Load Mistral LLM from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 500}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain


def answer_question(query: str):
    """
    Answers a user question using the RAG pipeline.
    :param query: User's question
    :return: Answer string
    """
    qa_chain = build_qa_chain()
    result = qa_chain({"query": query})
    return result["result"]


if __name__ == "__main__":
    question = "What are the revenue highlights for Q2?"
    answer = answer_question(question)
    print("Answer:", answer)
