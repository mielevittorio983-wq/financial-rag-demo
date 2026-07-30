from __future__ import annotations

import http.client
import json
import os
import re
from urllib.parse import urlsplit


BASE_URL = os.getenv(
    "BASE_URL",
    f"http://0.0.0.0:{os.getenv('PORT', '7860')}",
).rstrip("/")


def request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict, bytes]:
    target = urlsplit(BASE_URL)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=60)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"} if body else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    content = response.read()
    result_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, result_headers, content


def main() -> None:
    status, headers, _ = request("GET", "/")
    assert status == 307 and headers.get("location") == "/?demo=1"

    status, _, page = request("GET", "/?demo=1")
    page_text = page.decode("utf-8")
    assert status == 200 and "银行监管可信知识助手" in page_text

    assets = re.findall(r'(?:src|href)="(/static/[^"]+)"', page_text)
    assert assets
    for asset in assets:
        asset_status, _, asset_body = request("GET", asset)
        assert asset_status == 200 and asset_body

    status, _, health_body = request("GET", "/health")
    health = json.loads(health_body)
    assert status == 200
    assert health["status"] == "PASS"
    assert health["public_mode"] is True
    assert health["llm_mode"] == "DISABLED"

    status, _, presets_body = request("GET", "/demo/presets")
    presets = json.loads(presets_body)["presets"]
    scenario_statuses: dict[str, str] = {}
    for preset_id in ("regulation", "excel", "cross_file", "refusal"):
        preset = next(item for item in presets if item["id"] == preset_id)
        status, _, answer_body = request(
            "POST",
            "/answer",
            {"question": preset["question"], "requested_mode": preset["mode"]},
        )
        answer = json.loads(answer_body)
        assert status == 200, f"{preset_id}: HTTP {status}"
        assert answer["status"] in {"ANSWERED", "REFUSED"}
        scenario_statuses[preset_id] = answer["status"]
        assert answer["llm_mode"] == "DISABLED"
        if preset_id == "regulation":
            assert answer["status"] == "ANSWERED"
            assert answer["evidence"]
        if preset_id == "excel" and answer["status"] == "ANSWERED":
            assert any(item["coordinate"] for item in answer["evidence"])
        if preset_id == "cross_file" and answer["status"] == "ANSWERED":
            assert len(answer["evidence"]) >= 2
        if preset_id == "refusal":
            assert answer["status"] == "REFUSED"

    status, _, new_document_body = request("POST", "/demo/new-document")
    new_document = json.loads(new_document_body)
    assert status == 200
    assert new_document["status"] == "ANSWERED"
    assert "731" in new_document["answer"]
    assert new_document["evidence"]

    print("PASS: root redirect, demo page, static assets, UTF-8, health and retrieval")
    print("scenario_statuses=" + json.dumps(scenario_statuses, ensure_ascii=False, sort_keys=True))
    print("new_document=ANSWERED")


if __name__ == "__main__":
    main()
