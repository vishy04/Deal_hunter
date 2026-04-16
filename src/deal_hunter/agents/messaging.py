# Imports
from deal_hunter.agents.agent import Agent
from deal_hunter.services.notifications import PushoverNotifier
from deal_hunter.config import settings
from deal_hunter.models.deals import Opportunity


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(
        self,
        notifier: PushoverNotifier | None = None,
        model: str | None = None,
    ):
        self.model = model if model is not None else settings.messenger_model

        if notifier is None:
            self.notifier = PushoverNotifier(
                settings.pushover_user,
                settings.pushover_token,
            )
        else:
            self.notifier = notifier

    def push(self, text: str) -> bool:
        return self.notifier.send(text)

    def alert(self, opportunity: Opportunity) -> bool:
        """
        Make an alert about the specified Opportunity
        """
        # Deal has description url and price from deals.py datamodel
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f} :"
        text += opportunity.deal.product_description[:80] + "... "
        text += opportunity.deal.url
        self.log("Messaging agent is sending")
        ok = self.push(text)
        self.log("Messaging Agent is Completed")
        return ok

    def _fallback_craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        text = f"Deal Alert! Price={deal_price}, "
        text += f"Estimate=${estimated_true_value}, "
        text += f"Discount=${estimated_true_value - deal_price:.2f} :"
        text += description
        return text

    def craft_message(
        self, description: str, deal_price: float, estimated_true_value: float
    ) -> str:
        system_prompt = (
            "You write concise, exciting push notifications about product deals"
        )
        user_prompt = "Please Summarize this great deal in 2-3 sentences to be sent alerting the user about the deal.\n"
        user_prompt += f"Item Description: {description}\nOffered Price:{deal_price}\nEstimated True Value:{estimated_true_value}\n"
        user_prompt += "\n\n Respond only with the 2-3 sentence message which will be used to alert & excite the user about the deal "
        # Lazy import so `MessagingAgent` loads even when litellm is broken at install time;
        # failures surface only when this method runs.
        from litellm import completion

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as e:
            self.log(f"Calling LLM failed :{e}")
            return self._fallback_craft_message(
                description, deal_price, estimated_true_value
            )

        raw = response.choices[0].message.content
        if raw is None or not str(raw).strip():
            self.log("LLM returned empty message; using fallback text")
            return self._fallback_craft_message(
                description, deal_price, estimated_true_value
            )
        return str(raw).strip()

    def notify(self, description, deal_price, estimated_true_value, url) -> bool:
        self.log("Messaging Agent is crafting the message")
        text = self.craft_message(description, deal_price, estimated_true_value)
        text = text[:200] + "..." + url
        ok = self.push(text)
        self.log("Messaging Agent has completed")
        return ok
