from deal_hunter.agents.agent import Agent
from deal_hunter.agents.specialist import SpecialistAgent
from deal_hunter.agents.frontier import FrontierAgent
from deal_hunter.config import settings
from deal_hunter.services.preprocessing import Preprocessor


class EnsembleAgent(Agent):
    name = "Ensemble Agent"
    color = Agent.YELLOW

    def __init__(self, collection) -> None:
        self.log("Starting Ensemble Agent")

        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.preprocessor = Preprocessor()

        # ensemble weights
        self.frontier_weights = settings.ensemble_frontier_weights
        self.specialist_weights = settings.ensemble_specialist_weight
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        self.log("Running Ensemble Agent- Preprocessing Text")
        rewrite = self.preprocessor.preprocess(description)
        self.log(f"Preprocessing Text using:{self.preprocessor.model_name}")
        specialist = self.specialist.price(rewrite)
        frontier = self.frontier.price(rewrite)
        combined = frontier * (self.frontier_weights) + specialist * (
            self.specialist_weights
        )
        self.log(f"Ensemble Agent completed - Predicted ${combined:,.2f}")
        return combined
