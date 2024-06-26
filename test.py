import os
from operator import itemgetter
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

vector_store = PGVector(
    collection_name="collection164",
        connection_string = "postgresql+psycopg2://alok:Alok1234@localhost:5432/database164",

    embedding_function=OpenAIEmbeddings()
)

template = """
Answer given the following context:
{context}

Question: {question}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model='gpt-4-1106-preview', streaming=True)


class RagInput(TypedDict):
    question: str


final_chain = (
    {
    "context": (itemgetter("question") | vector_store.as_retriever()),
    "question": itemgetter("question")
    }
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)