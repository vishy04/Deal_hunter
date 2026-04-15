import requests
from deal_hunter.agents.agent import Agent
from typing import Any


class PushoverNotifier(Agent):
    name = "Pushover Notifier"
    color = Agent.MAGENTA

    def __init__(
        self,
        user_key: str,
        token: str,
        url: str = "https://api.pushover.net/1/messages.json",
    ) -> None:
        self.user_key = user_key
        self.token = token
        self.url = url

    def send(self, message: str, sound: str = "cashregister") -> bool:
        if len(message) > 1024:
            self.log("Message over pushover limit")

        payload = {
            "user": self.user_key,
            "token": self.token,
            "message": message,
            "sound": sound,
        }
        try:
            response = requests.post(self.url, data=payload, timeout=10)
            response.raise_for_status()  # throws an error for HTTP failures
        except Exception as e:
            self.log(f"Error getting HTTP: {e} ")
            return False  # end it
        try:
            data: dict[str, Any] = response.json()
        except Exception as e:
            self.log(f"Error getting json response : {e}")
            return False  # end it

        # check if we get any data
        if data.get("status") != 1:
            errors = data.get("errors", [])
            self.log(f"Pushover reject message errors = {errors}")
            return False

        # if no false then message pushed
        self.log("Pushover notification sent!")
        return True
