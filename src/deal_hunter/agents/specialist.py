import modal
from deal_hunter.agents.agent import Agent


class SpecialistAgent(Agent):
    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        self.log("Specialist Agent -- Connecting to Modal")
        Pricer = modal.Cls.from_name("pricer", "Pricer")
        self.pricer = Pricer()

    def price(self, description: str) -> float:
        self.log("Specialist Agent calling remote Finetuned Model")
        result = self.pricer.price.remote(description)

        self.log(f"Specialist Agent finished {result:.2f}")
        return result
