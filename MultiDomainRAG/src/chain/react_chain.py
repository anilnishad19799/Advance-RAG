# src/chains/react_chain.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class ReactAnswerGenerator:
    """
    ReAct-style LLM Chain for final answer synthesis from context + question.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.prompt_template = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template=(
                "{system_prompt}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Final Answer:"
            ),
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

        # domain-specific system prompts
        self.system_prompts = {
            "medical": (
                "You are an expert medical assistant. "
                "Answer user medical questions accurately using the provided context. "
                "If the question is unrelated to medicine, politely decline."
            ),
            "legal": (
                "You are a legal case summarization expert. "
                "Use the given case law or legal context to form a clear and factual answer. "
                "Do not provide legal advice or opinion, only summarize from context."
            ),
            "other": (
                "You are a general knowledge assistant. "
                "Use the provided text context to answer the question clearly."
            ),
        }

    def generate_answer(self, domain: str, context: str, question: str) -> str:
        """
        Generate a final answer using the domain-specific system prompt.
        """
        system_prompt = self.system_prompts.get(domain, self.system_prompts["other"])

        response = self.chain.run(
            system_prompt=system_prompt,
            context=context[:15000],  # truncate long contexts
            question=question,
        )

        return response.strip()
