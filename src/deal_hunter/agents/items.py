from typing import Self
from pydantic import BaseModel
from datasets import Dataset , DatasetDict , load_dataset 


class Item(BaseModel):


    title : str
    price : float
    category : str
    test_prompt : str
    
    @classmethod
    def from_hub(cls , dataset_name : str) -> tuple[list[Self] , list[Self], list[Self]]:
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["test"]],
            [cls.model_validate(row) for row in ds["val"]],
        )