from typing import List

from gptinference.openai_api import OpenaiAPIWrapper
from gptinference.utils import newline, shorten
from gptinference.caching import Caching, OpenAICacheKey, OpenAICacheValue
from typing import Dict

def cost_in_dollars(num_input_tokens: int, num_output_tokens: int, engine: str) -> float:
    """Calculate the dollar cost of a completion in dollars.
       See the pricing page for more details: https://openai.com/api/pricing/
    """
    if engine == "gpt-3.5-turbo":
        # GPT-3.5-turbo costs $0.50 / 1M input tokens and $1.50 / 1M output tokens
        return (num_input_tokens * 0.5e-6) + (num_output_tokens * 1.5e-6)
    elif engine == "gpt-4":
        # GPT-4 costs $30 / 1M input tokens and $60 / 1M output tokens
        return (num_input_tokens * 30e-6) + (num_output_tokens * 60e-6)
    elif engine == "gpt-4-turbo":
        # GPT-4-turbo costs $10 / 1M input tokens and $30 / 1M output tokens
        return (num_input_tokens * 10e-6) + (num_output_tokens * 30e-6)
    elif engine == "gpt-4o":
        # GPT-4-turbo costs $5 / 1M input tokens and $15 / 1M output tokens
        return (num_input_tokens * 5e-6) + (num_output_tokens * 15e-6)
    else:
        raise ValueError(f"Pricing unavailable for requested engine: {engine}")


class OpenAIWrapper:
        def __init__(self, cache_path:str=None, save_every_n_seconds: int=600):
            self.cache = Caching(cache_path=cache_path, save_every_n_seconds=save_every_n_seconds)

        def call(self, prompt, engine, max_tokens=300, stop_token="###", temperature=0.0, cost_estimator_info_to_fill: Dict=None):
            if not prompt:
                return ""
            cache_key = OpenAICacheKey(engine=engine,
                                       prompt=str(prompt).lstrip(), # don't store new lines in the beginning.
                                       stop_token=stop_token,
                                       temperature=temperature,
                                       max_tokens=max_tokens)
            cache_val = self.cache.get(key=cache_key)
            if not cache_val:
                # print(f"\nCalling GPT3: {shorten(prompt, max_words=10)}...")
                val_dict = OpenaiAPIWrapper.call(prompt=prompt,
                                                 engine=engine,
                                                 max_tokens=max_tokens,
                                                 stop_token=stop_token,
                                                 temperature=temperature)
                cache_val = self.cache.set(key=cache_key, value=OpenAICacheValue(
                    first_response=str(OpenaiAPIWrapper.get_first_response(response=val_dict, engine=engine))
                ))

                # val_dict contains the usage information:
                # "usage": {
                #     "completion_tokens": 17,
                #     "prompt_tokens": 57,
                #     "total_tokens": 74
                #   }
                # fill the cost estimator info with the dollar cost of the API call (since it was a cache miss).
                cost_estimator_info_to_fill["cost_in_dollars"] = { 
                    "dollar_cost": cost_in_dollars(
                                                num_input_tokens=val_dict.usage.prompt_tokens,
                                                num_output_tokens=val_dict.usage.completion_tokens,
                                                engine=engine
                                                ),
                    "input_tokens": val_dict.usage.prompt_tokens,
                    "output_tokens": val_dict.usage.completion_tokens
                }
            else:
                # This is a cache hit so we don't need to call the API and can just return the cached value without any cost.
                cost_estimator_info_to_fill["cost_in_dollars"] = { 
                    "dollar_cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }

            return cache_val.first_response

        def mk_cache_key(self, prompt: str, engine: str, max_tokens=300, stop_token="###", temperature=0.0) -> str:
            cache_key = OpenAICacheKey(engine=engine,
                                  prompt=prompt.lstrip(), # don't store new lines in the beginning.
                                  stop_token=stop_token,
                                  temperature=temperature,
                                  max_tokens=max_tokens)
            return cache_key


        def call_batch(self, prompts: List[str], engine: str, max_tokens=300, stop_token="###", temperature=0.0):
            cache_keys = [self.mk_cache_key(prompt=prompt, engine=engine, stop_token=stop_token, temperature=temperature, max_tokens=max_tokens)  for prompt in prompts]
            precached_entries = {prompt: self.cache.get(key=cache_key) for cache_key, prompt in zip(cache_keys, prompts)}
            uncached_prompts   = [prompt for prompt, cached_entry in precached_entries.items() if not cached_entry]

            # print(f"\nCalling GPT3 as a batch for ({len(uncached_prompts)}/ {len(precached_entries)}) "
            #       f"prompts: {(newline+newline).join([shorten(p, max_words=10)+ '...' for p in uncached_prompts])}")

            batched_response = OpenaiAPIWrapper.call(prompt=uncached_prompts,
                                                        engine=engine,
                                                        max_tokens=max_tokens, stop_token=stop_token,
                                                        temperature=temperature)

            for prompt, completion in zip(uncached_prompts, OpenaiAPIWrapper.get_first_response_batched(response=batched_response, engine=engine)):
                value = OpenAICacheValue(first_response= completion.strip())
                precached_entries[prompt] = value
                cache_key = self.mk_cache_key(prompt=prompt, engine=engine, stop_token=stop_token, temperature=temperature, max_tokens=max_tokens)
                self.cache.set(key=cache_key, value=value)

            return [p.first_response for _, p in precached_entries.items()]

