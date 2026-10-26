"""Scripted and HTTP LLM backends with a shared normalized response format."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

_RETRY_BACKOFF_SECONDS = (10, 30, 60)


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float = 600,
    on_retry=None,
):
    encoded = json.dumps(payload).encode()
    call = request.Request(url, data=encoded, headers=headers, method="POST")
    for retry_number in range(len(_RETRY_BACKOFF_SECONDS) + 1):
        retry_error = None
        try:
            with request.urlopen(call, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            if not 500 <= error.code < 600:
                raise
            retry_error = error
        except (TimeoutError, URLError, ConnectionError) as error:
            retry_error = error

        if retry_number == len(_RETRY_BACKOFF_SECONDS):
            raise retry_error
        if on_retry is not None:
            on_retry(
                {
                    "event": "http_retry",
                    "attempt": retry_number + 1,
                    "error": f"{type(retry_error).__name__}: {retry_error}",
                }
            )
        time.sleep(_RETRY_BACKOFF_SECONDS[retry_number])


class Backend:
    """Return one normalized assistant turn and token usage."""

    def set_event_handler(self, handler):
        self.event_handler = handler

    def _request_json(self, url, headers, payload):
        if self.requester is not None:
            return self.requester(url, headers, payload)
        return _post_json(
            url,
            headers,
            payload,
            timeout=self.request_timeout,
            on_retry=getattr(self, "event_handler", None),
        )

    def respond(self, messages, tools):
        raise NotImplementedError


class ScriptedBackend(Backend):
    """Privileged deterministic reference that follows the saved assembly order."""

    def __init__(self, plan: dict[str, Any], execution_order=None):
        self.plan = plan
        self.order = list(
            plan.get("assembly_order", [])
            if execution_order is None
            else execution_order
        )
        self.step = 0

    def respond(self, messages, tools):
        del messages, tools
        if self.step == 0:
            call = {"id": "script-0", "name": "get_task", "arguments": {}}
        elif self.step <= len(self.order):
            part = int(self.order[self.step - 1])
            x, y, z = map(float, self.plan["target"][part])
            call = {
                "id": f"script-{self.step}",
                "name": "move_part",
                "arguments": {
                    "part": part,
                    "x": x,
                    "y": y,
                    "z": z,
                    "yaw_deg": 0.0,
                },
            }
        else:
            call = {"id": f"script-{self.step}", "name": "submit", "arguments": {}}
        self.step += 1
        return {"content": "", "tool_calls": [call]}, {
            "input_tokens": 0,
            "output_tokens": 0,
        }


class AnthropicBackend(Backend):
    """Anthropic Messages API backend using native tool-use blocks."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.anthropic.com",
        requester=None,
        request_timeout: float = 600,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.requester = requester
        self.request_timeout = request_timeout

    @staticmethod
    def _serialize(messages):
        system = "\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        output = []
        for message in messages:
            role = message["role"]
            if role == "system":
                continue
            if role == "assistant":
                content = []
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": call["arguments"],
                    }
                    for call in message.get("tool_calls", [])
                )
                output.append({"role": "assistant", "content": content})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message["content"],
                }
                if (
                    output
                    and output[-1]["role"] == "user"
                    and isinstance(output[-1]["content"], list)
                ):
                    output[-1]["content"].append(block)
                else:
                    output.append({"role": "user", "content": [block]})
            else:
                output.append({"role": "user", "content": message["content"]})
        return system, output

    def respond(self, messages, tools):
        system, serialized = self._serialize(messages)
        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "system": system,
            "messages": serialized,
            "tools": [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ],
        }
        data = self._request_json(
            f"{self.base_url}/v1/messages",
            {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload,
        )
        turn = {"content": "", "tool_calls": []}
        texts = []
        for block in data.get("content", []):
            if block["type"] == "text":
                texts.append(block["text"])
            elif block["type"] == "tool_use":
                turn["tool_calls"].append(
                    {
                        "id": block["id"],
                        "name": block["name"],
                        "arguments": block.get("input", {}),
                    }
                )
        turn["content"] = "\n".join(texts)
        usage = data.get("usage", {})
        return turn, {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }


class OpenAIBackend(Backend):
    """OpenAI-compatible chat-completions backend, including local vLLM."""

    supports_images = True

    def __init__(
        self,
        model: str,
        base_url: str,
        requester=None,
        request_timeout: float = 600,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.requester = requester
        self.request_timeout = request_timeout

    @staticmethod
    def _serialize(messages):
        output = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                serialized_content = []
                for part in content:
                    if part.get("type") != "image_url":
                        serialized_content.append(part)
                        continue
                    image_url = dict(part["image_url"])
                    url = image_url.get("url")
                    if not isinstance(url, str):
                        raise TypeError("image_url references must contain a URL")
                    if url.startswith(("data:", "http://", "https://")):
                        serialized_content.append({**part, "image_url": image_url})
                        continue
                    image_path = Path(url.removeprefix("file://"))
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            f"image reference does not exist: {image_path}"
                        )
                    mime_type = (
                        mimetypes.guess_type(image_path.name)[0]
                        or "application/octet-stream"
                    )
                    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                    image_url["url"] = f"data:{mime_type};base64,{encoded}"
                    serialized_content.append({**part, "image_url": image_url})
                content = serialized_content
            item = {"role": message["role"], "content": content}
            if message["role"] == "assistant" and message.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in message["tool_calls"]
                ]
            if message["role"] == "tool":
                item["tool_call_id"] = message["tool_call_id"]
            output.append(item)
        return output

    def respond(self, messages, tools):
        data = self._request_json(
            f"{self.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'local')}",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "messages": self._serialize(messages),
                "tools": tools,
                "tool_choice": "auto",
            },
        )
        message = data["choices"][0]["message"]
        turn = {"content": message.get("content") or "", "tool_calls": []}
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments", "{}")
            turn["tool_calls"].append(
                {
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": json.loads(arguments) if arguments else {},
                }
            )
        usage = data.get("usage", {})
        return turn, {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }
