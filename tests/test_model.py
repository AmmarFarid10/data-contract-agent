import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from app.core.model import LocalModel, ModelError


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str


def response(body):
    return httpx.Response(200, json={"message": {"content": body}})


def test_model_retries_transient_errors_then_validates_schema():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503)
        return response('{"category":"shipping"}')

    model = LocalModel(transport=httpx.MockTransport(handler), sleeper=lambda _: None)
    assert model.structured(Output, "Classify", {"message": "untrusted"}).category == "shipping"
    assert len(calls) == 3


def test_timeout_retries_are_bounded():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic timeout")

    with pytest.raises(ModelError):
        LocalModel(transport=httpx.MockTransport(handler), sleeper=lambda _: None).structured(
            Output, "x", {}
        )
    assert len(calls) == 3


@pytest.mark.parametrize("body", ["not json", '{"category":"returns","execute":"shell"}', "{}"])
def test_invalid_model_json_fails_closed_without_retry(body):
    calls = []

    def handler(request):
        calls.append(request)
        return response(body)

    with pytest.raises(ModelError):
        LocalModel(transport=httpx.MockTransport(handler)).structured(Output, "x", {})
    assert len(calls) == 1


def test_arbitrary_model_endpoints_are_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://169.254.169.254")
    with pytest.raises(ValueError):
        LocalModel()
