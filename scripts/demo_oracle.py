"""Source-bound numerical assertions for the currently selected demo controls."""

import math
import re

_CACHE = None


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
    global _CACHE
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
