"""Verify every recorded preset, exact replay states and independent live math."""

import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from sampling_lab.artifacts import sha256, write_json
from sampling_lab.samplers import leapfrog
from sampling_lab.targets import Target


def main():
    checked, live, mobile, errors = [], [], [], []
    manifest = json.loads(Path("docs/explorer_manifest.json").read_text())
    with sync_playwright() as r:
        browser = r.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport=dict(width=1380, height=1024))
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(Path("docs/explorer.html").resolve().as_uri())
        page.get_by_role("tab", name="Sampler mechanics", exact=True).click()
        for rho in [0, 0.5, 0.8, 0.98]:
            for eta in [0.001, 0.02, 0.05, 0.5]:
                page.locator("#rho").fill(str(rho))
                page.locator("#eta").fill(str(eta))
                value = page.evaluate("window.liveMechanics")
                lam = np.linalg.eigvalsh([[1, rho], [rho, 1]])[::-1]
                np.testing.assert_allclose(value["targetEigen"], lam, atol=1e-12)
                assert value["stable"] == (eta < 2 * lam[-1])
                if value["stable"]:
                    np.testing.assert_allclose(
                        value["ulaEigen"], lam / (1 - eta / (2 * lam)), atol=1e-12
                    )
                else:
                    assert value["ulaEigen"] is None
                live.append(dict(rho=rho, eta=eta))
        gaussian = Target("gaussian", 1, np.zeros(1), np.eye(1))
        for h in [0.01, 0.1, 0.5]:
            for steps in [1, 2, 20, 100]:
                page.locator("#h").fill(str(h))
                page.locator("#steps").fill(str(steps))
                value = page.evaluate("window.liveMechanics")
                x, p, valid = leapfrog(
                    gaussian, np.array([0.8]), np.array([0.3]), h, steps, np.eye(1)
                )
                np.testing.assert_allclose([value["x"], value["p"]], [x[0], -p[0]], atol=2e-14)
                expected = 0.5 * (float(x[0]) ** 2 + float(p[0]) ** 2)
                np.testing.assert_allclose(value["energy"], expected, atol=2e-14)
                np.testing.assert_allclose(
                    value["acceptance"], np.exp(min(0, 0.365 - expected)), atol=2e-14
                )
                assert valid
                live.append(dict(h=h, steps=steps))
        for selector, value in [("#rho", "0.8"), ("#eta", "0.05"), ("#h", "0.1"), ("#steps", "20")]:
            page.locator(selector).fill(value)
        page.screenshot(path="assets/previews/mechanics_desktop.png", full_page=True)
        for group in page.locator("nav button").all_text_contents():
            page.get_by_role("tab", name=group, exact=True).click()
            if group != "Sampler mechanics":
                for task in page.locator("#task option").all_text_contents():
                    page.select_option("#task", task)
                    for preset in page.locator("#preset option").all_text_contents():
                        page.select_option("#preset", preset)
                        page.wait_for_function(
                            'document.querySelector("#figure").complete && document.querySelector("#figure").naturalWidth>0'
                        )
                        assert page.locator("#title").inner_text() == preset
                        if group != "Energy-based learning":
                            for draw in [1, 500, 2000]:
                                page.locator("#draw").fill(str(draw))
                                actual = page.evaluate("window.replayEvidence")
                                with np.load(
                                    Path("assets/source_data") / (actual["name"] + ".npz"),
                                    allow_pickle=False,
                                ) as data:
                                    np.testing.assert_array_equal(
                                        actual["points"], data["observables"][:, draw - 1, :2]
                                    )
                            for chain in ["all", "0", "1", "2", "3"]:
                                page.select_option("#chain", chain)
                            page.select_option("#chain", "all")
                        checked.append(dict(group=group, task=task, preset=preset))
            page.set_viewport_size(dict(width=390, height=844))
            assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), group
            mobile.append(group)
            page.set_viewport_size(dict(width=1380, height=1024))
        page.get_by_role("tab", name="Geometry and reparameterization", exact=True).click()
        page.select_option("#task", "funnel_noncentered")
        page.select_option("#preset", "nuts · dispersed · preset 0.1")
        page.wait_for_function(
            'document.querySelector("#figure").complete && document.querySelector("#figure").naturalWidth>0'
        )
        page.screenshot(path="assets/previews/funnel_desktop.png", full_page=True)
        page.get_by_role("tab", name="Energy-based learning", exact=True).click()
        page.select_option("#task", "mixture")
        page.select_option("#preset", "quadrature_mle · fit seed 42")
        page.wait_for_function(
            'document.querySelector("#figure").complete && document.querySelector("#figure").naturalWidth>0'
        )
        page.screenshot(path="assets/previews/energy_desktop.png", full_page=True)
        for concept, task, preset, filename in [
            (
                "Chain diagnostics",
                "gaussian",
                "gibbs · dispersed · preset 1",
                "sampling_gaussian.png",
            ),
            (
                "Multimodality and tempering",
                "ring",
                "parallel_tempering · dispersed · preset 0.6",
                "sampling_ring.png",
            ),
            (
                "Geometry and reparameterization",
                "banana",
                "hmc · dispersed · preset 0.12",
                "sampling_banana.png",
            ),
        ]:
            page.get_by_role("tab", name=concept, exact=True).click()
            page.select_option("#task", task)
            page.select_option("#preset", preset)
            page.select_option("#sample-view", "cloud")
            page.select_option("#plot-window", "focus")
            page.locator("#draw").fill("2000")
            page.screenshot(path="assets/previews/" + filename, full_page=True)
        page.get_by_role("tab", name="Energy-based learning", exact=True).click()
        page.set_viewport_size(dict(width=390, height=844))
        page.screenshot(path="assets/previews/explorer_mobile.png", full_page=True)
        browser.close()
    assert not errors, errors
    assert len(checked) == manifest["presets"]
    write_json(
        "docs/explorer_browser_validation.json",
        dict(
            status="passed",
            html_sha256=sha256("docs/explorer.html"),
            presets=checked,
            live_checks=live,
            mobile_concepts=mobile,
            javascript_errors=errors,
            exact_replay_checks=84 * 3,
        ),
    )
    print(
        "Passed",
        len(checked),
        "presets;",
        len(live),
        "live calculations;",
        84 * 3,
        "exact replay states",
    )


if __name__ == "__main__":
    main()
