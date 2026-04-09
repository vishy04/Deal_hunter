from deal_hunter.agents.agent import Agent
from deal_hunter.agents.specialist import SpecialistAgent
from deal_hunter.agents.frontier import FrontierAgent
from deal_hunter.services.preprocessing import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection) -> None:
        self.log("Starting Ensemble Agent")

        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.preprocessor = Preprocessor()

        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        self.log("Running Ensemble Agent- Preprocessing Text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Preprocessing Text using:{self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        combined = frontier * (0.8115124189175806) + specialist * (0.1884875810824194)
        self.log(f"Ensemble Agent completed - Predicted ${combined:,.2f}")
        return combined
