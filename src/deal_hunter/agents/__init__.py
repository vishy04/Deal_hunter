def __getattr__(name: str):
    if name == "ScannerAgent":
        from deal_hunter.agents.scanner import ScannerAgent
        return ScannerAgent
    if name == "MessagingAgent":
        from deal_hunter.agents.messaging import MessagingAgent
        return MessagingAgent
    if name == "PlanningAgent":
        from deal_hunter.agents.planner import PlanningAgent
        return PlanningAgent
    if name == "AutonomousPlanner":
        from deal_hunter.agents.autonomous_planner import AutonomousPlanner
        return AutonomousPlanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
