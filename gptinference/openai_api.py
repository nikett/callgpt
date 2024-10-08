import os
from typing import Dict, Any, List, Union
import openai
import random
import time

# Use the latest openai chat /v1/ endpoint.
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or openai.api_key)

# check if org is set (never needed it, so commenting out)
# if os.getenv("OPENAI_ORG") is not None:
#     openai.organization = os.getenv("OPENAI_ORG")
MAX_TRIES= int(os.getenv("OPENAI_MAX_TRIES_INT")) if os.getenv("OPENAI_MAX_TRIES_INT") is not None else 10

# from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = MAX_TRIES,
    errors: tuple = (openai.RateLimitError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def is_chat_based_agent(engine):
    return not engine.lower().strip() == "gpt-3"

class OpenaiAPIWrapper:
    @staticmethod
    @retry_with_exponential_backoff
    def call(
        prompt: Union[str, List[str], List[Dict[str, str]]],
        max_tokens: int,
        engine: str,
        stop_token: str,
        temperature: float,
        num_completions: int = 1
    ) -> dict:
        
        if is_chat_based_agent(engine): # gpt 3.5 onwards.
            # check if batched requests (list of text prompts) are requested.
            batched_requested = isinstance(prompt, List) and len(prompt) > 1 and isinstance(prompt[0], str)
            assert not (batched_requested and is_chat_based_agent(engine)), \
                f"Open AI does not support batched requests. Check your prompt in the call to OpenaiAPIWrapper."
            
            if isinstance(prompt, List):
                assert len(prompt) >= 1, f"No prompt given as input to call OpenAI API."

            # check if prompt is a list of strings or a list of dictionaries
            # gpt-3.5-turbo onwards does not support a batched list of prompts.
            # but, the conversation API does support a list of dictionaries.
            conversational_content = [{"role": "user", "content": prompt[0]}] if isinstance(prompt, List) and isinstance(prompt[0], str) \
                        else ([{"role": "user", "content": prompt}] if isinstance(prompt, str) \
                        else prompt)
            response = openai_client.chat.completions.create(
                model=engine,
                messages=conversational_content,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stop=[stop_token],
                n=num_completions
            )
        else:  # chatgpt onwards.
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stop=[stop_token],
                # logprobs=3,
                n=num_completions
            )

        return response

    @staticmethod
    def get_first_response(response, engine) -> Dict[str, Any]:
        """Returns the first response from the list of responses.
        Sample response:
        {
        "choices": [
            {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                "role": "assistant"
            },
            "logprobs": null
            }
        ],
        "created": 1677664795,
        "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
        "model": "gpt-3.5-turbo-0613",
        "object": "chat.completion",
        "usage": {
            "completion_tokens": 17,
            "prompt_tokens": 57,
            "total_tokens": 74
        }
        }
        """
        if is_chat_based_agent(engine):
            text = response.choices[0].message.content
        else:
            text = response["choices"][0]["text"]
        return text

    @staticmethod
    def get_first_response_batched(response, engine) -> List[Dict[str, Any]]:
        """Returns the first response from the list of responses."""
        if is_chat_based_agent(engine):
            for r in response["choices"]:
                yield r.message.content
        else:
            for r in response["choices"]:
                yield r["text"]

    # @staticmethod
    # def get_all_responses(response, engine) -> Dict[str, Any]:
    #     """Returns the list of responses."""
    #     return [choice["text"] for choice in response["choices"]]  # type: ignore
