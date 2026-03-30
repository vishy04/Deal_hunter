import logging

class Agent:

    """
    Used for better logging so that each agent can be identified
    """
        # Foreground colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background color
    BG_BLACK = '\033[40m'
    
    # Reset code to return to default color
    RESET = '\033[0m'

    name: str = ""
    color: str = '\033[37m'

    def log(self,message):
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)