import json
from types import SimpleNamespace

import pytest

from tv_voice.bridge import Screen
from tv_voice.model import GeminiPlanner


async def test_request_contains_only_current_screen_and_bounded_structured_output(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    planner = GeminiPlanner("gemini-3.5-flash-lite")
    requests = []

    async def generate_content(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            text='{"kind":"press","button":"right","message":"Moving right.","observation":"Home."}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=100, candidates_token_count=20, thoughts_token_count=0
            ),
        )

    monkeypatch.setattr(planner.client.aio.models, "generate_content", generate_content)
    try:
        decision = await planner.decide("Open Netflix", Screen(b"pixels", "a", 0, 720, 540), [])
        assert decision.button == "right"
        assert requests[0]["config"].max_output_tokens == 768
        assert requests[0]["config"].response_mime_type == "application/json"
        assert requests[0]["contents"][0].parts[1].inline_data.data == b"pixels"
        assert (planner.input_tokens, planner.output_tokens) == (100, 20)
    finally:
        await planner.close()


def test_missing_api_key_fails_before_any_model_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiPlanner("gemini-3.5-flash-lite")


async def test_real_sdk_serializes_schema_and_image_without_network(monkeypatch):
    import httpx2
    from google import genai

    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": '{"kind":"ask","message":"Which app?","observation":"Home is visible."}'
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20},
            },
        )

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    factory = genai.Client

    def make_client(**kwargs):
        kwargs["http_options"] = kwargs["http_options"].model_copy(
            update={"httpx_async_client": http}
        )
        return factory(**kwargs)

    monkeypatch.setattr(genai, "Client", make_client)
    planner = GeminiPlanner("gemini-3.5-flash-lite")
    try:
        result = await planner.decide("Open it", Screen(b"pixels", "a", 0, 720, 540), [])
        assert result.kind == "ask"
        assert len(requests) == 1
        config = requests[0]["generationConfig"]
        # ProtoJSON accepts both original proto names and lowerCamelCase names.
        thinking = config["thinkingConfig"]
        assert thinking.get("thinkingLevel", thinking.get("thinking_level")) == "MINIMAL"
        assert config["responseMimeType"] == "application/json"
        assert requests[0]["contents"][0]["parts"][1]["inlineData"]["data"] == "cGl4ZWxz"
    finally:
        await planner.close()
        await http.aclose()
