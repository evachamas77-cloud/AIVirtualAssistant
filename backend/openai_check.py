"""Check OpenAI model metadata without generating an answer or sending an image."""

import argparse
import asyncio
import os
import re
from urllib.parse import quote

import httpx

from .vision import _api_key, _provider, _provider_diagnostics, _provider_response_metadata


async def _compare_model_lookup(model: str, key: str) -> None:
    print("SDK routing settings (values hidden): " + ", ".join(
        f"{name}={'set' if name in os.environ else 'unset'}"
        for name in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID")
    ))
    url = f"https://api.openai.com/v1/models/{quote(model, safe='')}"
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        for label, token in (("configured key", key), ("dummy key", "deliberately-invalid-diagnostic-token")):
            try:
                async with asyncio.timeout(15):
                    response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
            except (httpx.TimeoutException, TimeoutError):
                result = "timed out"
            except (httpx.HTTPError, ValueError):
                result = "could not complete the request"
            else:
                body = "present" if response.content else "empty"
                result = f"HTTP {response.status_code} body={body}"
            print(f"Minimal HTTP lookup ({label}): {result}")


async def check_model(model: str | None, *, compare: bool = False) -> int:
    if model == "":
        print("No model configured. Set AI_VISION_MODEL or pass --model.")
        return 1
    if model is not None and model.lower().startswith("sk-"):
        print("The model value looks like a key. Put credentials only in OPENAI_API_KEY.")
        return 1
    key = _api_key()
    if not key:
        print("No API key in this terminal. Set OPENAI_API_KEY with the masked prompt in README.")
        return 1
    if not all(33 <= ord(char) <= 126 for char in key) or key.startswith(("'", '"')) or key.endswith(("'", '"')):
        print("OPENAI_API_KEY contains spaces, quotes around the key, or invalid characters. Re-enter just the key.")
        return 1

    from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

    operation = "model list" if model is None else "model lookup"
    try:
        async with _provider(key, 15) as client:
            async with asyncio.timeout(15):
                if model is None:
                    page = await client.models.list()
                else:
                    await client.models.retrieve(model)
    except APIStatusError as error:
        kind, code, param, message = _provider_diagnostics(error)
        body, request_id = _provider_response_metadata(error)
        print(
            f"OpenAI {operation}: HTTP {error.status_code} type={kind} code={code} "
            f"param={param} body={body} request_id={request_id}. {message}"
        )
        if error.status_code == 400 and compare and model is not None:
            await _compare_model_lookup(model, key)
        elif error.status_code == 401:
            print("Check that the replacement key is active and belongs to your project.")
        elif error.status_code == 403:
            print("Check whether this key is permitted to read model metadata.")
        elif error.status_code == 404 and model is not None:
            print("This model could not be retrieved. Run --list to check which models are visible.")
        return 1
    except (APITimeoutError, TimeoutError):
        print(f"OpenAI {operation} timed out.")
        return 1
    except APIConnectionError:
        print(f"Could not connect to OpenAI for the {operation}.")
        return 1
    except APIError:
        print(f"OpenAI returned an invalid {operation} response.")
        return 1

    if model is None:
        data = getattr(page, "data", None)
        if not isinstance(data, list):
            print("OpenAI returned an invalid model list response.")
            return 1
        model_ids = (getattr(entry, "id", None) for entry in data)
        visible = sorted({
            name for name in model_ids
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name)
            and not name.lower().startswith("sk-") and name != key
        })
        print("Available model IDs (metadata only; not all models support images):")
        print("\n".join(visible) if visible else "(none listed)")
    else:
        print("OpenAI model lookup succeeded. This checks model metadata, not vision generation.")
    print("No question or image was sent.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--model", help="Look up this model instead of AI_VISION_MODEL")
    mode.add_argument("--list", action="store_true", help="List model IDs visible to this key")
    parser.add_argument("--compare", action="store_true", help="Compare two minimal GETs if the SDK returns HTTP 400")
    args = parser.parse_args()
    if args.list:
        if args.compare:
            parser.error("--compare requires a single model lookup")
        return asyncio.run(check_model(None))
    model = args.model if args.model is not None else os.getenv("AI_VISION_MODEL", "gpt-4o-mini")
    return asyncio.run(check_model(model.strip(), compare=args.compare))


if __name__ == "__main__":
    raise SystemExit(main())
