# RSS service: decisions and learnings

## The problem

The original version put all scraping logic inside `ScrapedDeal.__init__`. Every time you constructed a deal object, it fired off `requests.get()`, parsed HTML, and stored the result. One 404 killed the whole batch because there was no error handling. Feed URLs were hardcoded at module level. You couldn't test anything without a network connection.

I already separated `ScrapedDeal` into a plain `BaseModel` in the data models refactor. The HTTP and parsing logic still needed somewhere to go. That's `services/rss.py`.

## What the service does

`Rss_Service` extends `Agent` for colored log output. Three methods.

### extract()

Takes raw HTML from the RSS `<summary>` field, finds `div.snippet_summary`, strips tags, collapses whitespace. Falls back to stripping the entire snippet if that div is missing. Same algorithm as before, just living in a service now instead of a data model.

### scrape_entry()

Turns one feedparser entry dict into a `ScrapedDeal`. It pulls the title, runs `extract()` on the summary HTML, grabs the URL from `entry.links[0]["href"]` (not `entry.url`, which is always None), then fetches the product page. If the page has a `div.content-section`, it splits on "Features" to separate details from feature lists. Then it builds a `ScrapedDeal` and calls `truncate()`.

When the page fetch fails or the HTML is different from what we expect, it logs the error and returns a deal with just RSS-level data. Details and features stay empty. The old code would crash here instead.

### scrape_feeds()

Takes a list of feed URLs, parses each with feedparser, grabs up to `max_per_feed` entries per feed, calls `scrape_entry()` on each. Sleeps `delay` seconds between requests so we don't hammer DealNews.

`known_urls` handles deduplication. Pass in URLs you've already processed and the service skips them. Originally this filtering lived inside `ScannerAgent.fetch_deals()`. Moving it here means the agent doesn't need to think about deduplication.

## Error handling

Log and skip, don't crash. Two layers to this.

If `feedparser.parse()` throws on one feed URL, the service logs it and moves to the next feed. The rest still get scraped. If one entry's product page is down or returns unexpected HTML, `scrape_entry()` catches the exception and returns whatever it got from RSS. If even that fails, `scrape_feeds()` catches it at the outer level and moves on.

Logs go through `Agent.log()` with a cyan `[RSS_Service]` prefix.

## Why a class

I originally planned standalone functions. Went with a class because extending `Agent` gives colored, name-prefixed logging without any extra work. The cost is one line of instantiation (`rss = Rss_Service()`).

## The URL field gotcha

feedparser entries have `entry.link` (a string) and `entry.links` (a list of dicts). There is no `entry.url`. The actual URL is in `entry.links[0]["href"]`, with `entry.link` as a fallback.

An early version of the code did `entry.get("links", [{}])` without indexing into the list. That passed a list object where a URL string was expected. It didn't blow up until `requests.get()` tried to use it.

## Where it fits

```
settings.rss_feed_url -> ScannerAgent -> Rss_Service.scrape_feeds() -> [ScrapedDeal, ...]
```

ScannerAgent reads feed URLs from config, passes them to the service, gets back `ScrapedDeal` instances. The agent calls `deal.describe()` on each and sends the text to GPT for structured extraction into `Deal` objects.
