import inspect
import json
import logging
from copy import deepcopy
from typing import List, Literal, Union

import docstring_parser
from pii_utils import replace_pii
from openai import (
    APITimeoutError,
    ConflictError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)
from pydantic import BaseModel, create_model
from tenacity import retry, stop_after_attempt, wait_random_exponential

DEFAULT_MODEL = "gpt-4o"


LlmModelLiteral = Literal["gpt-4o", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-turbo"]


def clean_message(message):
    message = deepcopy(message)
    message["content"] = replace_pii(message["content"])
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            tool_call["function"]["arguments"] = replace_pii(
                tool_call["function"]["arguments"]
            )
    return message


def should_retry(retry_state):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        for error_type in [
            RateLimitError,
            UnprocessableEntityError,
            APITimeoutError,
            ConflictError,
        ]:
            if isinstance(exception, error_type):
                return True
    return False


class LlmApi:
    def __init__(self):
        self.client = None

    def init_api(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def init_app(self, app):
        api_key = (
            app.config["ODIN_NOSTRA_SERVICE_KEY"]
            if app.config["ENV"] != "local"
            else app.config["OPENAI_API_KEY"]
        )
        self.client = OpenAI(
            api_key=api_key, base_url=app.config["OPENAI_API_BASE_URL"]
        )

    def singleton():
        if not hasattr(LlmApi, "_instance"):
            LlmApi._instance = LlmApi()
        return LlmApi._instance

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        # retry if not successful and not auth error
        retry=should_retry,
    )
    def get_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)


class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)


def query_llm(
    messages: Union[str, List[dict]],
    system_prompt: str = None,
    model: str = None,
    tools=None,
    tool_choice="auto",
    temperature: float = 0.0,
    verbose: bool = True,
    max_turns: int = 3,
    response_model: BaseModel = None,
    exec=True,
    force_json=False,
):
    """
    Query the LLM with a list of messages.

    :param force_json: Force response to be JSON. Only needed if response_model is not provided. In your system prompt make sure to include instructions to output in JSON, or else the response might hang.
    """

    # print params with truncation
    if not model:
        model = DEFAULT_MODEL
    if isinstance(messages, str):  # messages can be string or list of messages
        messages = [{"role": "user", "content": messages}]
    if system_prompt:
        # remove prior system prompt
        messages = [m for m in messages if m.get("role") != "system"]
        # Inserting system_prompt at end of messages seems to be more reliable than inserting at beginning
        # messages.insert(0, {"role": "system", "content": system_prompt})
        if force_json and "json" not in system_prompt.lower():
            system_prompt += "\n\nPlease always output in JSON format."
        messages.append({"role": "system", "content": system_prompt})
    if response_model:
        tools = [get_formal_openai_schema(response_model)]
        tool_choice = dict(
            type="function", function=dict(name=tools[0]["function"]["name"])
        )
    elif tools:
        name_to_function = {tool.__name__: tool for tool in tools if callable(tool)}
        tools = [get_formal_openai_schema(tool) for tool in tools]
        if (
            tool_choice
            and not isinstance(tool_choice, dict)
            and tool_choice not in ["auto", "required", "none"]
        ):
            if callable(tool_choice):
                tool_choice = tool_choice.__name__
            tool_choice = dict(type="function", function=dict(name=tool_choice))
    else:
        name_to_function = {}
        tool_choice = None
    if verbose:
        print(
            f">>> [LLM INPUT] model={model} tool_choice={tool_choice} tools={tools} system_prompt={str(system_prompt)[:200]}...  messages={str(messages)[:200]}..."
        )

    num_turns = 0
    messages = [
        message for message in messages if message["role"] != "status"
    ]  # any dropping of messages must happen BEFORE init_message_len is set
    init_message_len = len(messages)
    while num_turns < max_turns:
        messages = [clean_message(message) for message in messages]
        kwargs = dict(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            model=model,
            temperature=temperature,
        )
        if force_json:
            kwargs["response_format"] = dict(type="json_object")
        completion = LlmApi.singleton().get_completion(**kwargs)
        response_message = completion.choices[0].message.to_dict()
        if verbose:
            print(f"<<< [LLM OUTPUT] {response_message}")
        messages.append(response_message)
        if response_model:
            return response_model.parse_obj(
                json.loads(response_message["tool_calls"][0]["function"]["arguments"])
            )
        elif response_message.get("tool_calls") and exec:
            num_turns += 1
            for tool_call in response_message["tool_calls"]:
                function = tool_call["function"]
                function_name = function["name"]
                function_args = json.loads(function["arguments"])
                func = name_to_function.get(function_name)
                if not func:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "tool_name": function_name,
                            "content": f"Error: Tool {function_name} not found.",
                        }
                    )
                    continue
                logging.info(
                    f"Calling function {function_name} with args {function_args}"
                )
                results = func(**function_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function["name"],
                        "content": json.dumps(results, cls=PydanticEncoder),
                        "args": function_args,
                    }
                )
            # always reset tool_choice to auto so we can attempt to return a human-readable message
            tool_choice = "auto"
        elif force_json:
            return json.loads(response_message["content"])
        else:
            break
    return messages[init_message_len:]


def get_formal_openai_schema(tool):
    if isinstance(tool, dict):
        return dict(type="function", function=tool)
    if callable(tool):
        return dict(type="function", function=get_schema(tool)[0])
    raise Exception(f"Invalid tool: {tool}")


def get_schema(thing, name=None, description=""):
    """
    Convert function or class (example: Pydantic model) to OpenAPI schema.

    See: https://github.com/jxnl/instructor/blob/cd5169e8b6d263220bf9a8a29a8235f7268ea02e/instructor/function_calls.py#L30-L50

    :param thing: function or class
    :param name: Overrides name
    :param description: Overrides description
    """
    if not name:
        name = thing.__name__
    if inspect.isclass(thing) and issubclass(thing, BaseModel):
        model = thing
    else:
        signature = inspect.signature(thing)
        name_to_info = {}
        for param_name, param_info in signature.parameters.items():
            if param_info.annotation is inspect.Parameter.empty:
                raise Exception(f"Missing type hint for parameter {param_name}")
            name_to_info[param_name] = (
                param_info.annotation,
                (
                    param_info.default
                    if param_info.default is not inspect.Parameter.empty
                    else None
                ),
            )
        model = create_model(name, __base__=BaseModel, **name_to_info)
    properties = model.schema()["properties"]
    if thing.__doc__:
        doc_string = docstring_parser.parse(thing.__doc__)
        for doc in doc_string.params:
            parameter = properties.get(doc.arg_name)
            if parameter and doc.description and "description" not in parameter:
                parameter["description"] = doc.description
        if not description and doc_string.description:
            description = doc_string.description.strip()
    required = [k for k, v in properties.items() if v.get("default") is None]
    deep_delete_key(properties, "title")
    return (
        dict(
            name=name,
            description=description,
            parameters=dict(type="object", properties=properties, required=required),
        ),
        model,
    )


def deep_delete_key(data, key):
    if isinstance(data, dict):
        for k in list(data.keys()):
            if k == key and "type" in data:
                del data[k]
            else:
                deep_delete_key(data[k], key)
