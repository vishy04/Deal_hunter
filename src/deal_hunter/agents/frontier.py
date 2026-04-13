import re
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from deal_hunter.agents.agent import Agent


class FrontierAgent(Agent):
    name = "FrontierAgent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"

    def __init__(self, collection) -> None:
        self.log("Starting Frontier Agent")
        self.client = OpenAI()

        self.MODEL = "gpt-40-mini"

        self.collection = collection
        self.encoder_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        message = "Here are the similar products with prices, which can be helpful for context\n"
        for similar, price in zip(similars, prices):
            message += f"Related Product:\n{similar}\nPrice is ${price}\n\n"
        return message

    def message_for(
        self, description: str, similars: List[str], prices: List[float]
    ) -> List[Dict[str, str]]:
        message = f"Estimate the price of the product. Respond with Price only No Explanation\n\n{description}"
        message += self.make_context(similars, prices)
        return [{"role": "user", "content": message}]

    def find_similar(self, text: str):
        self.log(
            "Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products"
        )
        vector = self.encoder_model.encode(text)
        result = self.collection.query(
            query_embeddings=vector,
            include=["documents", "metadatas"],
            n_results=5,
        )
        document = result["documents"][0][:]
        price = [m["price"] for m in result["metadatas"][0][:]]

        self.log("Frontier Agent has found Similar Products")

        return document, price

    def get_price(self, text) -> float:
        text = text.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        return float(match.group()) if match else 0

    def price(self, description: str) -> float:
        documents, prices = self.find_similar(description)
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.message_for(description, documents, prices),
            seed=42,
            reasoning_effort="none",
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent Completed - predicted {result:.2f}")
        return result
