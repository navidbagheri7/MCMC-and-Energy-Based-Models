"""Source-bound numerical assertions for the currently selected demo controls."""

import math
import re
from pathlib import Path

import numpy as np

_CACHE = None
_SAMPLES = {}
_TARGETS = None


def close(text, expected):
    if expected is None:
        assert text == "—", (text, expected)
        return
    if isinstance(expected, str):
        assert text == expected, (text, expected)
        return
    found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
    assert found, (text, expected)
    token = found.group().replace(",", "")
    places = len(token.split("e")[0].partition(".")[2])
    exponent = int(token.lower().partition("e")[2] or "0")
    tolerance = 0.501 * 10 ** (exponent - places)
    assert math.isclose(float(token), expected, rel_tol=1e-12, abs_tol=tolerance), (text, expected)


def cards(page, selector, expected):
    actual = page.locator(selector).all_text_contents()
    assert len(actual) == len(expected), (actual, expected)
    for text, value in zip(actual, expected, strict=True):
        close(text, value)


def image_sources(page, selector, expected):
    actual = page.locator(selector).evaluate_all('xs=>xs.map(x=>x.getAttribute("src"))')
    assert actual == expected, (actual, expected)


def check(page):
    global _CACHE, _TARGETS
    if _CACHE is None:
        _CACHE = page.evaluate("evidence.rows.map(({points,...r})=>r)")
    if not page.locator("#preset:visible").count():
        return
    group = page.locator('nav [aria-selected="true"]').inner_text()
    task = page.locator("#task").input_value()
    preset = page.locator("#preset").input_value()
    r = next(
        r for r in _CACHE if r["group"] == group and r["task"] == task and r["preset"] == preset
    )
    m = r["metrics"]
    image_sources(page, "#figure", [r["image"]])
    assert page.locator("#source").get_attribute("href") == r["source"]
    if r["kind"] == "sampling":
        if _TARGETS is None:
            _TARGETS = page.evaluate("evidence.targets")
        if r["name"] not in _SAMPLES:
            path = Path(__file__).resolve().parents[1] / "assets/source_data" / (r["name"] + ".npz")
            with np.load(path, allow_pickle=False) as source:
                _SAMPLES[r["name"]] = np.array(source["observables"][:, :, :2])
        source = _SAMPLES[r["name"]]
        actual = page.evaluate("window.sampleViewEvidence")
        draw = int(page.locator("#draw").input_value())
        chain = page.locator("#chain").input_value()
        indices = list(range(4)) if chain == "all" else [int(chain)]
        bounds = np.asarray(_TARGETS[task]["bounds"])
        if page.locator("#plot-window").input_value() == "full":
            lo, hi = source.min(axis=(0, 1)), source.max(axis=(0, 1))
            pad = np.maximum(hi - lo, 1e-6) * 0.05
            bounds = np.stack([lo - pad, hi + pad], axis=1)
        np.testing.assert_allclose(actual["bounds"], bounds, rtol=1e-12, atol=1e-12)
        points = source[indices, :draw]
        outside = np.any((points < bounds[:, 0]) | (points > bounds[:, 1]), axis=2).sum()
        assert actual["outside"] == outside
        assert actual["displayed"] == len(indices) * draw
        assert (
            actual["name"] == r["name"]
            and actual["view"] == page.locator("#sample-view").input_value()
        )
        for axis, histogram in enumerate(actual["histograms"]):
            edges = np.linspace(*bounds[axis], 41)
            for index, bar in zip(indices, histogram["chains"], strict=True):
                count, _ = np.histogram(source[index, :draw, axis], edges)
                assert bar["chain"] == index
                np.testing.assert_array_equal(bar["counts"], count)
                np.testing.assert_allclose(
                    bar["density"], count / (draw * np.diff(edges)), rtol=1e-11, atol=1e-12
                )
        d = m["diagnostics"]

        def maximum(xs):
            return max(x for x in xs if x is not None)

        def minimum(xs):
            return min(x for x in xs if x is not None)

        expected = [
            maximum(d["rank_rhat"]),
            minimum(d["bulk_ess"]),
            minimum(d["tail_ess"]),
            d["divergences"],
            "Passed" if d["convergence_flag"] else "Failed",
            d["total_sampling_and_warmup_seconds"],
        ]
    else:
        expected = [
            m["selected_step"],
            m["validation_nll"],
            m["test_nll"],
            m["partition_max_change"],
            m["negative_gradient_evaluations"],
            m["fitting_seconds"],
        ]
    cards(page, "#numbers .number", expected)
