import modal 

from modal import Image

app = modal.App("Hello")
image = Image.debian_slim().pip_install("requests")

@app.function(image=image)
def hello() -> str:
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()