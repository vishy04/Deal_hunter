from pydantic import BaseModel, Field
from typing import List


class ScrapedDeal(BaseModel):
    title: str
    summary: str
    url: str
    details: str = ""
    features: str = ""

    def truncate(self):
        self.title = self.title[:100]
        self.details = self.details[:500]
        self.features = self.features[:500]

    def __repr__(self) -> str:
        return f"<{self.title}>"

    def describe(self):
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL:{self.url}"


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
