"""HTTP client for the churn API. The frontend's only route to predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

ErrorKind = Literal["unavailable", "not_ready", "invalid", "server"]


@dataclass
class ApiError(Exception):
    kind: ErrorKind
    message: str
    field_errors: list[dict[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.message


class ChurnApiClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise ApiError(
                "unavailable", f"The API at {self.base_url} did not respond in time."
            ) from exc
        except httpx.TransportError as exc:
            raise ApiError("unavailable", f"Could not reach the API at {self.base_url}.") from exc

        if response.status_code == 422:
            body = _json_or_empty(response)
            raise ApiError(
                "invalid",
                body.get("detail", "The API rejected the request."),
                field_errors=body.get("errors", []),
            )
        if response.status_code == 503:
            body = _json_or_empty(response)
            raise ApiError("not_ready", body.get("detail", "The model is not ready yet."))
        if response.status_code >= 400:
            raise ApiError(
                "server", f"The API returned an unexpected error (HTTP {response.status_code})."
            )
        return _json_or_empty(response)

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=payload)

    def predict_batch(
        self,
        payloads: list[dict[str, Any]],
        max_batch_size: int = 100,
        exploration: bool = True,
    ) -> list[float]:
        """Probabilities for many payloads, split into requests the API will accept.

        exploration=True marks the rows as hypothetical what-if scenarios so the API keeps
        them out of drift monitoring.
        """
        headers = {"X-Churn-Purpose": "exploration"} if exploration else None
        probabilities: list[float] = []
        for start in range(0, len(payloads), max_batch_size):
            chunk = payloads[start : start + max_batch_size]
            body = self._request(
                "POST", "/predict/batch", json={"customers": chunk}, headers=headers
            )
            probabilities += [p["churn_probability"] for p in body["predictions"]]
        return probabilities


def _json_or_empty(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
