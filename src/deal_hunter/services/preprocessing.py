from litellm import completion

from dotenv import load_dotenv
import os

load_dotenv(override = True)


SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""
DEFAULT_MODEL_NAME = "groq/openai/gpt-oss-20b"
DEFAULT_REASONING_EFFORT = "low"
class Preprocessor:

    def __init__(self,model_name = DEFAULT_MODEL_NAME, reasoning_effort = "low") -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def messages_for(self,text:str) -> list[dict] :
        return   [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},]

    def preprocess(self , text: str) -> str:
        messages = self.messages_for(text)
        response = completion(
            model = self.model_name,
            messages=messages,
            reasoning_effort = self.reasoning_effort,
        )
        return response.choices[0].message.content