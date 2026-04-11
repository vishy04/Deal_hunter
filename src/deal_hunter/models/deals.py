from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
import re
from typing import List, Dict, Self
import feedparser
from tqdm import tqdm
import sys
from pathlib import Path
import time

root_dir = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(root_dir / "src"))

from deal_hunter.config import settings


feeds = settings.rss_feed_url


def extract(html_snippet: str) -> str:
    soup = BeautifulSoup(html_snippet, "html.parser")
    snippet_div = soup.find("div", class_="snippet summary")

    if snippet_div:
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, "html.parser").get_text()
        description = re.sub("<[^<]+?>", "", description)
        result = description.strip()
    else:
        result = html_snippet
    return result.replace("\n", " ")


class ScrapedDeal:
    title: str
    summary: str
    url: str
    details: str = ""
    features: str = ""

    def __init__(self, entry: Dict[str, str]) -> None:
        self.title = entry["title"]
        self.summary = extract(entry["summary"])
        self.url = entry["links"][0]["href"]
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, "html.parser")
        content = soup.find("div", class_="content-section").get_text()
        if "Features" in content:
            self.details, self.features = content.split("Features", 1)
        else:
            self.details = content
            self.features = ""
        self.truncate()

    def truncate(self):
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self):
        return f"Title:{self.title}\nDetails:{self.details.strip()}\n Features: {self.features.strip()}\nURL:{self.url}"

    @classmethod
    def fetch(cls, show_progess: bool = False) -> List[Self]:
        deals = []
        feed_itr = tqdm(feeds) if show_progess else feeds

        for feed_url in feed_itr:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                deals.append(cls(entry))
                time.sleep(0.05)

        return deals


class Deal(BaseModel):
    product_description: str = Field(
        description=(
            "A 3-4 sentence summary focused on the product itself: "
            "brand, model name, key specs (size, capacity, wattage, etc.), "
            "and primary use case. Do NOT mention discounts, coupons, or "
            "savings — describe only what the buyer receives."
        )
    )
    price: float = Field(
        description=(
            "The final out-of-pocket price in USD after all advertised "
            "discounts, coupons, and rebates are applied. If the deal "
            "states '$100 off $300', return 200.0. If a price range is "
            "given, use the lowest advertised price. Use 0.0 for free items."
        )
    )

    url: str = Field(
        description=(
            "The full URL linking to this deal, exactly as it appeared in "
            "the input. Do not modify, shorten, or infer URLs."
        )
    )


class DealSelection(BaseModel):
    deals: List[Deal] = Field(
        description=(
            "Select exactly 5 deals from the input that best satisfy ALL of "
            "these criteria, in priority order: "
            "(1) the listing contains enough detail to write a concrete "
            "product description (brand, model, specs); "
            "(2) the price is clearly stated or unambiguously calculable; "
            "(3) the deal represents genuine value — not a minor accessory "
            "or a vague 'up to X% off' promotion with no specific product."
        )
    )


class Opportunity(BaseModel):
    deal: Deal
    estimate: float
    discount: float
