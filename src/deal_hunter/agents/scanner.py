from deal_hunter.agents.agent import Agent
from deal_hunter.config import settings
from deal_hunter.models.deals import DealSelection, ScrapedDeal
from deal_hunter.services.rss import Rss_Service
from openai import OpenAI


class ScannerAgent(Agent):
    name = "Scanner Agent"
    color = Agent.CYAN

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    """

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a short paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    def __init__(
        self, rss: Rss_Service | None = None, openai_client: OpenAI | None = None
    ):
        self.feed_urls = settings.rss_feed_url
        self.scanner_model = settings.scanner_model
        self.rss = rss or Rss_Service()
        self.openai = openai_client or OpenAI()

        self.log("Scanner Agent Initialized")

    def fetch_deals(self, memory: list[str]) -> list[ScrapedDeal]:
        known_urls = set(memory or [])
        self.log(f"Fetching RSS Deals : {len(known_urls)}")

        scraped = self.rss.scrape_feeds(
            feed_urls=self.feed_urls,
            known_urls=known_urls,
        )

        self.log(f"Fetched {len(scraped)} new deals")

        return scraped

    def make_user_prompt(self, scraped: list[ScrapedDeal]) -> str:
        deal_text = "\n\n".join(deal.describe() for deal in scraped)
        return f"{self.USER_PROMPT_PREFIX}{deal_text}{self.USER_PROMPT_SUFFIX}"

    def scan(self, memory: list[str] | None = None) -> DealSelection | None:
        """
        fetch deals
        send to openai for structured output
        filter invalid prices output from openai
        """

        scraped = self.fetch_deals(memory=memory)
        if not scraped:
            self.log("No deals found")
            return None

        user_prompt = self.make_user_prompt(scraped=scraped)

        completion = self.openai.chat.completions.parse(
            model=self.scanner_model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=DealSelection,
            reasoning_effort="minimal",
        )
        parsed = completion.choices[0].message.parsed
        # filter 0 price deals where modal cannot get clear price
        parsed.deals = [deal for deal in parsed.deals if deal.price > 0]

        return parsed

    def test_scan(self) -> DealSelection:
        # sample test data for testing
        results = {
            "deals": [
                {
                    "product_description": "The Hisense R6 Series 55R6030N is a 55-inch 4K UHD Roku Smart TV that offers 3840x2160 resolution, Dolby Vision HDR, and HDR10 support. It uses Roku OS for easy streaming app access and voice assistant compatibility. The TV also includes three HDMI ports for connecting multiple devices.",
                    "price": 178,
                    "url": "https://www.dealnews.com/products/Hisense/Hisense-R6-Series-55-R6030-N-55-4-K-UHD-Roku-Smart-TV/484824.html?iref=rss-c142",
                },
                {
                    "product_description": "The Poly Studio P21 is a 21.5-inch 1080p personal meeting display designed for remote work. It includes a built-in 1080p webcam, stereo speakers, privacy shutter, and adjustable lighting features. It also supports wireless charging for compatible devices.",
                    "price": 30,
                    "url": "https://www.dealnews.com/products/Poly-Studio-P21-21-5-1080-p-LED-Personal-Meeting-Display/378335.html?iref=rss-c39",
                },
                {
                    "product_description": "The Lenovo IdeaPad Slim 5 features an AMD Ryzen 5 8645HS processor, 16GB RAM, and 512GB SSD storage. Its 16-inch 1080p touch display offers clear visuals for productivity and everyday tasks. This hardware mix gives a good balance of speed and usability.",
                    "price": 446,
                    "url": "https://www.dealnews.com/products/Lenovo/Lenovo-Idea-Pad-Slim-5-7-th-Gen-Ryzen-5-16-Touch-Laptop/485068.html?iref=rss-c39",
                },
                {
                    "product_description": "The Dell G15 gaming laptop includes an AMD Ryzen 5 7640HS CPU, NVIDIA RTX 3050 GPU, 16GB RAM, and a 1TB NVMe SSD. It has a 15.6-inch 1080p display with a 120Hz refresh rate for smoother gameplay. The configuration is suitable for gaming and heavier workloads.",
                    "price": 650,
                    "url": "https://www.dealnews.com/products/Dell/Dell-G15-Ryzen-5-15-6-Gaming-Laptop-w-Nvidia-RTX-3050/485067.html?iref=rss-c39",
                },
            ]
        }
        return DealSelection.model_validate(results)
