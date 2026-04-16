# imports
from deal_hunter.config import settings
from deal_hunter.agents.agent import Agent
from deal_hunter.agents.scanner import ScannerAgent
from deal_hunter.agents.ensemble import EnsembleAgent
from deal_hunter.agents.messaging import MessagingAgent
from deal_hunter.models.deals import Deal, Opportunity


class PlanningAgent(Agent):
    name = "Planning Agent"
    color = Agent.GREEN

    def __init__(self, collection, scanner=None, ensemble=None, messenger=None):
        self.log("Planning Agent is Initialized")
        self.scanner = scanner or ScannerAgent()
        self.ensemble = ensemble or EnsembleAgent(collection)
        self.messenger = messenger or MessagingAgent()
        self.log("Planning Agent is Ready")

    def run(self, deal: Deal) -> Opportunity:
        """Takes a deal(basically deal info) -> turns to opportunity(potential deal)"""
        # will give the price using 5 similar in dataset(GPT+ RAG)+finetuned Llama on modal
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: list[Opportunity] | None = None) -> Opportunity | None:
        self.log("Planning Agent is starting a run")

        known_url = (
            [opp.deal.url for opp in memory] if memory else []
        )  # getting to url from list of opportunity datamodel

        # Scan RSS Feed for new deals
        selection = self.scanner.scan(memory=known_url)
        if not selection:
            self.log("No New Deals Found")
            return None

        # get price of every deal using run
        opportunities = [self.run(deal) for deal in selection.deals[:5]]

        # picking the highest discount deal
        opportunities.sort(key=lambda opp: opp.discount, reverse=True)

        best = opportunities[0]

        if best.discount > settings.deal_threshold:
            self.messenger.alert(best)
            self.log("Planning Agent Completed --deal sent")
            return best

        self.log("Planning Agent Completed --no deal above threshold")
        return None
