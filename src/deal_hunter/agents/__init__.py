def __getattr__(name: str):
    if name == "ScannerAgent":
        from deal_hunter.agents.scanner import ScannerAgent
        return ScannerAgent
    if name == "MessagingAgent":
        from deal_hunter.agents.messaging import MessagingAgent
        return MessagingAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
