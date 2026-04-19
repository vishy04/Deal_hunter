# imports
import chromadb
import json
from pathlib import Path
from deal_hunter.agents.agent import Agent
from deal_hunter.agents.planner import PlanningAgent
from deal_hunter.config import settings
from deal_hunter.models.deals import Opportunity


class DealAgentFramework(Agent):
    name = "Agent Framework"
    color = Agent.CYAN

    def __init__(self):
        """
        Open the chromaDB
        Load Memory
        lazy create planner -> creating is expensive
        """

        self.log("Starting Agent Framework")

        client = chromadb.PersistentClient(path=settings.vectorstore_path)
        self.collection = client.get_or_create_collection(
            settings.vectorstore_collection
        )

        self.memory = self._read_memory()
        self.planner = None

        self.log("Agent Framework is ready")

    def _read_memory(self) -> list[Opportunity]:
        path = Path(settings.memory_filename)

        if not path.exists():
            return []

        with open(path, "r") as f:
            data = json.load(f)

        return [Opportunity(**item) for item in data]

    def _write_memory(self) -> None:
        data = [opp.model_dump() for opp in self.memory]
        with open(settings.memory_filename, "w") as f:
            json.dump(data, f, indent=2)

    def _init_planner(self) -> None:
        if not self.planner:
            self.log("Starting planning Agent")
            self.planner = PlanningAgent(self.collection)
            self.log("Planning Agent Started")

    def run(self) -> list[Opportunity]:
        self._init_planner()
        self.log("Starting Run")
        result = self.planner.plan(memory=self.memory)

        if result:
            self.memory.append(result)
            self._write_memory()
            self.log(f"Run Complete - deal found with {result.discount:.2f} discount")
        else:
            self.log("Run Complete -- No deal Found")
        return self.memory

    @classmethod
    def reset_memory(cls) -> None:
        path = Path(settings.memory_filename)
        if not path.exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        truncated = data[:2]
        with open(path, "w") as f:
            json.dump(truncated, f, indent=2)


if __name__ == "__main__":
    DealAgentFramework().run()
