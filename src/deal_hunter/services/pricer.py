
import modal 
from modal import Image, Volume



#setup 

app = modal.App("pricer")
image = Image.debian_slim().pip_install(
    "torch","transformers","bitsandbytes","accelerate","peft"
)

secrets = [modal.Secret.from_name("huggingface-secret")]

#Constants
GPU = "A10G"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = "/cache"
MIN_CONTAINERS = 0

#prompt
PREFIX = "Price is $"
QUESTION = "How much does this cost to the nearest dollar"

#using Volume to store hf_download
hf_cache_volume = Volume.from_name("hf-hub-cache",create_if_missing = True)

@app.cls(
    image = image.env({"HF_HUB_CACHE":CACHE_DIR}),
    secrets = secrets,
    gpu = GPU , 
    timeout = 1800,
    min_containers = MIN_CONTAINERS ,
    volumes = {CACHE_DIR:hf_cache_volume},
)


class Pricer:

    #creating entry point to setup
    @modal.enter()
    def setup(self):
            #imports
            import torch

            from peft import PeftModel

            from transformers import (
                AutoModelForCausalLM, 
                BitsAndBytesConfig, 
                AutoTokenizer
            )

            #quant_config
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 ,
                bnb_4bit_quant_type="nf4" ,
            )

            #load_tokenizer
            #load->pad_token + side
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            self.tokenizer.pad_token = self.tokenizer.eos_token 
            self.tokenizer.padding_side = "right"


            #load model ( base ->finetuned )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, 
                quantization_config = quant_config,
                device_map = "auto",
            )
            self.fine_tuned_model = PeftModel.from_pretrained(
                self.base_model, "Vishy08/product-pricer-08-12-2025_04.35.08"
            )

    @modal.method()
    def price(self,description:str) -> float:
        #imports
        import re
        import torch
        from transformers import set_seed

        prompt = f"{QUESTION} \n\n{description}\n\n{PREFIX}"
        set_seed(42)
        #output -> decode -> price_extractor -> return 
        inputs = self.tokenizer.encode(prompt,return_tensors = "pt").to("cuda")

        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs , max_new_tokens = 5)
        
        result = self.tokenizer.decode(outputs[0])
        contents = result.split("Price is $")[1]
        contents = contents.replace(",","")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)

        return float(match.group()) if match else 0 

