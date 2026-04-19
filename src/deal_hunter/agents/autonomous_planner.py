# imports

import json
from openai import OpenAI
from deal_hunter.agents.scanner import ScannerAgent
from deal_hunter.agents.messaging import MessagingAgent
from deal_hunter.agents.ensemble import EnsembleAgent
from deal_hunter.agents.agent import Agent
from deal_hunter.config import settings
from deal_hunter.models.deals import Opportunity, Deal


class AutonomousPlanner(Agent):
    name = "Autonomous Agent"
    color = Agent.GREEN

    def __init__(
        self,
        collection,
        scanner: ScannerAgent | None = None,
        messaging: MessagingAgent | None = None,
        ensemble: EnsembleAgent | None = None,
        openai_client: OpenAI | None = None,
    ):
        self.log("Starting Autonomous Agent")
        self.scanner = scanner or ScannerAgent()
        self.ensemble = ensemble or EnsembleAgent(collection)
        self.messenger = messaging or MessagingAgent()
        self.openai = openai_client or OpenAI()
        self.model = settings.planner_model

        self.memory: list[str] = []
        self.opportunity: Opportunity | None = (
            None  # using this as a safety guard for messaging
        )
        self.log("Autonomous Agent is Ready")

    def deal_scanner(self) -> str:
        self.log("Autonomous Planning Agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No Deals Found"

    def estimate_value(self, description) -> str:
        self.log("Autonomous Planning Agent is estimating value using ensemble agent")
        estimate = self.ensemble.price(description=description)
        return f"The estimated Price of the {description} = {estimate}"

    def message_user(self, description, deal_price, estimated_true_value, url) -> str:
        if self.opportunity:
            self.log("Autonomous Planning Agent tried messaging twice (ignore)")
            return "Notification Already sent;Duplicate(Ignore)"
        self.log("Autonomous Planning Agent is notifying user")
        self.messenger.notify(description, deal_price, estimated_true_value, url)

        deal = Deal(product_description=description, price=deal_price, url=url)

        discount = estimated_true_value - deal.price
        self.opportunity = Opportunity(
            deal=deal, estimate=estimated_true_value, discount=discount
        )

        return "Notification Sent"

    # tool schemas
    scan_function = {
        "name": "deal_scanner",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "message_user",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item itself",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal",
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual value",
                },
                "url": {"type": "string", "description": "The URL of this deal"},
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self) -> list[dict]:
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tools(self, message) -> list[dict]:
        mapping = {
            "deal_scanner": self.deal_scanner,
            "estimate_value": self.estimate_value,
            "message_user": self.message_user,
        }

        results = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else "Unknown Tool"

            results.append(
                {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id,
                }
            )

        return results

    SYSTEM_MESSAGE = (
        "You find great deals on bargain products using your tools, "
        "and notify the user of the best bargain."
    )
    USER_MESSAGE = (
        "First, use your tool to scan the internet for bargain deals. "
        "Then for each deal, use your tool to estimate its true value. "
        "Then pick the single most compelling deal where the price is much "
        "lower than the estimated true value, and use your tool to notify the user. "
        "Then just reply OK to indicate success."
    )

    def plan(self, memory: list[Opportunity] | None = None) -> Opportunity | None:
        self.log("Autonomous Planning agent is starting a run")

        self.memory = [opp.deal.url for opp in memory] if memory else []

        self.opportunity = None

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": self.USER_MESSAGE},
        ]

        done = False

        while not done:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.get_tools(),
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)
                messages.extend(self.handle_tools(choice.message))
            else:
                done = True

        self.log(f"Autonomous Planning Agent completed:{choice.message.content}")
        return self.opportunity
