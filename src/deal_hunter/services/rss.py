# imports
import re
import requests

from bs4 import BeautifulSoup
from deal_hunter.agents.agent import Agent
from deal_hunter.models.deals import ScrapedDeal
import feedparser
import time


class Rss_Service(Agent):
    name = "RSS_Service"
    color = Agent.CYAN

    def extract(self, html_snippet: str) -> str:
        soup = BeautifulSoup(html_snippet, "html.parser")
        div = soup.find("div", class_="snippet_summary")
        text = (
            div.get_text(separator=" ", strip=True)
            if div
            else soup.get_text(separator=" ", strip=True)
        )
        text = re.sub("<[^<]+?>", "", text)
        result = text.strip().replace("\n", " ")

        return result

    def scrape_entry(self, entry: dict) -> ScrapedDeal:
        title = entry.get("title", "")
        summary = self.extract(entry.get("summary", ""))
        url = entry.get("links", [{}])[0].get("href", entry.get("link", ""))

        details = summary
        features = ""

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            content = soup.find("div", class_="content-section")
            if content:
                text = content.get_text(separator="\n")
                if "Features" in text:
                    parts = text.split("Features", 1)
                    details = parts[0].strip()
                    features = parts[1].strip()
                else:
                    details = text.strip()

        except (requests.RequestException, AttributeError, IndexError) as exc:
            self.log(f"failed to fetch product{url}: {exc}")

        deal = ScrapedDeal(
            title=title,
            summary=summary,
            url=url,
            details=details,
            features=features,
        )

        deal.truncate()

        return deal

    def scrape_feeds(
        self,
        feed_urls: list[str],
        known_urls: set[str] | None = None,
        max_per_feed: int = 10,
        delay: float = 0.05,
    ) -> list[ScrapedDeal]:
        if known_urls is None:
            known_urls = set()

        all_entries: list[dict] = []

        for feed_url in feed_urls:
            try:
                feed = feedparser.parse(feed_url)
                all_entries.extend(feed.entries[:max_per_feed])
            except Exception as e:
                self.log(f"Failed to parse feed: {feed_url}: {e}")

        deals = []

        for entry in all_entries:
            url = entry.get("links", [{}])[0].get("href", entry.get("link", ""))
            if url in known_urls:
                continue
            try:
                deals.append(self.scrape_entry(entry))
                known_urls.add(url)
            except Exception as e:
                self.log(f"Scrapping Failed for {entry.get('title', '?')}: {e}")
            time.sleep(delay)

        return deals
