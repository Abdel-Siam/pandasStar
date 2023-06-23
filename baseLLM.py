""" Base class to implement a new LLM

This module is the base class to integrate the various LLMs API. This module also includes the
Base LLM classes for OpenAI, HuggingFace and Google PaLM.

Example:

    ```
    from .base import BaseOpenAI

    class CustomLLM(BaseOpenAI):

        Custom Class Starts here!!
    ```
"""

import ast
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .constants import END_CODE_TAG, START_CODE_TAG
from .exceptions import (
    APIKeyNotFoundError,
    MethodNotImplementedError,
    NoCodeFoundError,
)
from .helpers._optional import import_optional_dependency
from .prompts.base import Prompt


class LLM:
    """Base class to implement a new LLM."""

    last_prompt: Optional[str] = None

    @property
    def type(self) -> str:
        """
        Return type of LLM.

        Raises:
            APIKeyNotFoundError: Type has not been implemented

        Returns:
            str: Type of LLM a string
        """
        raise APIKeyNotFoundError("Type has not been implemented")

    def _polish_code(self, code: str) -> str:
        """
        Polish the code by removing the leading "python" or "py",  \
        removing the imports and removing trailing spaces and new lines.

        Args:
            code (str): Code

        Returns:
            str: Polished code
        """
        if re.match(r"^(python|py)", code):
            code = re.sub(r"^(python|py)", "", code)
        if re.match(r"^`.*`$", code):
            code = re.sub(r"^`(.*)`$", r"\1", code)
        code = code.strip()
        return code

    def _is_python_code(self, string):
        """
        Return True if it is valid python code.
        Args:
            string (str):

        Returns (bool): True if Python Code otherwise False

        """
        try:
            ast.parse(string)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, response: str, separator: str = "```") -> str:
        """
        Extract the code from the response.

        Args:
            response (str): Response
            separator (str, optional): Separator. Defaults to "```".

        Raises:
            NoCodeFoundError: No code found in the response

        Returns:
            str: Extracted code from the response
        """
        
        # mainly for starcoder
        # responses = response.split('Code:')
        # responses = responses[1].split('Expected output:')
        # code = responses[0]
        
        code = response
        
        # print(START_CODE_TAG)
        # print(END_CODE_TAG)
        # matching for starcoder 
        # match = re.search(
        #     rf"{START_CODE_TAG}\n(.*)\n({END_CODE_TAG})",
        #     code,
        #     re.DOTALL,
        # )

        match = re.search(
            rf"```python(.*)```",
            code,
            re.DOTALL,
        )
        if match:
            # print(match)
            # print(f"group 0: {match.group(0).strip()}")
            # print(f"group 1: {match.group(1).strip()}")
            #print(f"group 2: {match.group(2).strip()}")
            code = match.group(1).strip()
            code = code.replace(START_CODE_TAG, '')
            code = code.replace(END_CODE_TAG, '')
            code = code.replace("""# Define the dataframe\ndf = pd.read_csv('path/to/file.csv')""", '')
            # for star coder
            # code = code.split(END_CODE_TAG)[0]
            # code = code.replace('</startCode>', '')
        if len(code.split(separator)) > 1:
            code = code.split(separator)[1]
            
        code = self._polish_code(code)
        if not self._is_python_code(code):
            raise NoCodeFoundError("No code found in the response")
        print(f"\nFinal Code: {code}")
        return code

    @abstractmethod
    def call(self, instruction: Prompt, value: str, suffix: str = "", responseLength = 512) -> str:
        """
        Execute the LLM with given prompt.

        Args:
            instruction (Prompt): Prompt
            value (str): Value
            suffix (str, optional): Suffix. Defaults to "".

        Raises:
            MethodNotImplementedError: Call method has not been implemented
        """
        raise MethodNotImplementedError("Call method has not been implemented")

    def generate_code(self, instruction: Prompt, prompt: str, responseLength = 512) -> str:
        """
        Generate the code based on the instruction and the given prompt.

        Returns:
            str: Code
        """
        return self._extract_code(self.call(instruction, prompt, suffix="\n\nCode:\n", responseLength = responseLength))

class HuggingFaceLLM(LLM):
    """Base class to implement a new Hugging Face LLM.

    LLM base class is extended to be used with HuggingFace LLM Modes APIs

    """

    """Starcoder LLM API

    A base HuggingFaceLLM class is extended to use Starcoder model.

    """
    _max_retries: int = 5
    last_prompt: Optional[str] = None
    
    def __init__(self):
        """
        __init__ method of Starcoder Class
        Args:
            api_token (str): API token from Huggingface platform
        """
        print("Initializing LLM ..... \n")
        from transformers import AutoTokenizer, pipeline, logging
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        
        # better in english
        #model_name_or_path = "TheBloke/starcoderplus-GPTQ"
        # better in coding
        # model_name_or_path = "TheBloke/starcoder-GPTQ"
        
        # model_basename = 'gptq_model-4bit--1g'
        # use_triton = False
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        # self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        #             model_basename=model_basename,
        #             use_safetensors=True,
        #             trust_remote_code=True,
        #             device="cuda:0",
        #             use_triton=use_triton,
        #             quantize_config=None)
        
        
        from transformers import AutoTokenizer, pipeline, logging
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        import argparse

        model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"
        # Or to load it locally, pass the local download path
        # model_name_or_path = "/path/to/models/TheBloke_WizardCoder-15B-1.0-GPTQ"

        use_triton = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                use_safetensors=True,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)

        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

    @property
    def type(self) -> str:
        return "starcoder"
    

    def query(self, payload, responseLength):

        #print(payload)


        # input_ids = self.tokenizer(payload, return_tensors='pt').input_ids.cuda()
        # output = self.model.generate(inputs=input_ids, max_length = responseLength, pad_token_id=self.tokenizer.eos_token_id)
        # print(self.tokenizer.decode(output[0]))
        from transformers import pipeline
        
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n\n### Response:"
        prompt = prompt_template.format(query=payload.replace('Code:', 'Assume that pandas has already been imported and that the dataframe is labeled as df. So DO NOT import pandas and DO NOT define df.'))
        # We use a special <|end|> token with ID 49155 to denote ends of a turn
        outputs = pipe(prompt, max_new_tokens=responseLength, do_sample=True, temperature=0.05, top_k=50, top_p=0.95, eos_token_id=49155)
        # You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.
        print(outputs[0]['generated_text'])
        return outputs[0]['generated_text']
        #return self.tokenizer.decode(output[0])

    def call(self, instruction: Prompt, value: str, suffix: str = "", responseLength = 512) -> str:
        """
        A call method of HuggingFaceLLM class.
        Args:
            instruction (object): A prompt object
            value (str):
            suffix (str):

        Returns (str): A string response

        """

        prompt = str(instruction)
        payload = prompt + value + suffix

        # sometimes the API doesn't return a valid response, so we retry passing the
        # output generated from the previous call as the input
        for _i in range(self._max_retries):
            response = self.query(payload, responseLength)
            payload = response
            break
            #changed so that after the first response it breaks and doesnt retry
            # if response.count("<endCode>") >= 2:
            #     break

        # replace instruction + value from the inputs to avoid showing it in the output
        output = response.replace(prompt + value + suffix, "")
        return output