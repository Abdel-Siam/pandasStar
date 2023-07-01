""" Starcoder LLM
This module is to run the StartCoder API hosted and maintained by HuggingFace.co.
To generate HF_TOKEN go to https://huggingface.co/settings/tokens after creating Account
on the platform.

Example:
    Use below example to call Starcoder Model

    >>> from pandasai.llm.starcoder import Starcoder
"""


import os
from typing import Optional

from dotenv import load_dotenv

from .base import SpecialBase

load_dotenv()


class WizardCoder(SpecialBase):

    """WizardCoder LLM API

    The current best free open source llm model
    """
    _max_retries: int = 5

    def __init__(self):
        from transformers import AutoTokenizer, pipeline, logging
        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

    def query(self, payload, responseLength: int = 512):
        """
        Query the HF API
        Args:
            payload: A JSON form payload

        Returns: Generated Response

        """
        from transformers import AutoTokenizer, pipeline, logging
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"

        print(f"Initializing {model_name_or_path} LLM ..... \n")
        use_triton = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                use_safetensors=True,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)

        from transformers import pipeline
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n\n### Response:"
        # here we take the original payload and modify it to so that it would fit in side of our template so that the model would answer it better. So the original payload ie the prompt in the file is modified a bit
        prompt = prompt_template.format(query=payload.replace('Code:', 'Assume that pandas has already been imported and that the dataframe is labeled as df. So DO NOT import pandas and DO NOT define df.'))
        # We use a special <|end|> token with ID 49155 to denote ends of a turn
        print(prompt)
        outputs = pipe(prompt, max_new_tokens=512, temperature=0.05, top_k=50, top_p=0.95, eos_token_id=49155)
        # You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.
        print(outputs[0]['generated_text'])
        return outputs[0]['generated_text']
    @property
    def type(self) -> str:
        return "WizardCoder"
