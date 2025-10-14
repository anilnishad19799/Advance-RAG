# src/utils/multi_query_generator.py

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import List


class MultiQueryGenerator:
    """
    Generates multiple search queries from a single input query
    using an LLM. Can be used for advanced RAG pipelines.
    """

    def __init__(self, num_queries: int = 3, temperature: float = 0.0):
        """
        Args:
            num_queries (int): Number of sub-queries to generate.
            temperature (float): Temperature for LLM generation.
        """
        self.num_queries = num_queries
        self.llm = ChatOpenAI(temperature=temperature)
        self.prompt_template = (
            "You are a helpful assistant that generates multiple search queries based on a single input query.\n"
            "Generate {num_queries} different search queries related to: {question}.\n"
            "Output only the queries, one per line."
        )
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.output_parser = StrOutputParser()

    def generate(self, query: str) -> List[str]:
        """
        Generate multiple sub-queries from a single query.

        Args:
            query (str): Original user query.

        Returns:
            List[str]: List of generated sub-queries.
        """
        formatted_prompt = self.prompt.format_prompt(
            question=query, num_queries=self.num_queries
        )
        raw_message = self.llm(
            formatted_prompt.to_messages()
        )  # returns AIMessage object
        raw_text = raw_message.content  # extract text from AIMessage
        # Parse output into list of queries
        queries = [
            q.strip()
            for q in self.output_parser.parse(raw_text).split("\n")
            if q.strip()
        ]
        return queries
