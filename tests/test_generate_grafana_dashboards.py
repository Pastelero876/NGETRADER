from __future__ import annotations

import json
import runpy


def test_generate_dashboard_outputs_json(capsys):
    runpy.run_path("scripts/generate_grafana_dashboards.py", run_name="__main__")
    out = capsys.readouterr().out
    data = json.loads(out)
    assert isinstance(data, dict) and data.get("title")
    titles = [p.get("title") for p in data.get("panels", [])]
    assert any("Latency" in str(t) for t in titles)


