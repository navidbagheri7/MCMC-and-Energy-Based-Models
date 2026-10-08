"use strict";
const evidence = JSON.parse(document.getElementById("evidence").textContent),
    groups = ["Sampler mechanics", "Chain diagnostics", "Multimodality and tempering", "Geometry and reparameterization", "Warm starts", "Bayesian regression", "Bayesian classification", "Energy-based learning"],
    colors = ["#087f8c", "#ac4b7f", "#c18317", "#4360a5"];
for (const row of evidence.rows) {
    if (row.kind !== "sampling") continue;
    const bytes = Uint8Array.from(atob(row.points_b64), c => c.charCodeAt(0));
    const view = new DataView(bytes.buffer);
    let offset = 0;
    row.points = Array.from({
        length: row.shape[0]
    }, () => Array.from({
        length: row.shape[1]
    }, () => [view.getFloat64((offset++) * 8, true), view.getFloat64((offset++) * 8, true)]));
    delete row.points_b64;
}
const $ = id => document.getElementById(id);
let group = groups[0],
    timer = null;

function node(tag, text) {
    let e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    return e;
}

function fmt(v) {
    return v === null || v === undefined ? "—" : typeof v === "number" ? (Math.abs(v) > 1e5 || Math.abs(v) < .001 && v !== 0 ? v.toExponential(3) : v.toFixed(3)) : String(v);
}

function extrema(xs, which) {
    return xs.filter(x => x !== null).reduce((a, b) => which === "max" ? Math.max(a, b) : Math.min(a, b), which === "max" ? -Infinity : Infinity);
}

function options(id, values) {
    $(id).replaceChildren(...values.map(v => {
        let e = node("option", v);
        e.value = v;
        return e;
    }));
}

function numbers(id, pairs) {
    $(id).replaceChildren(...pairs.map(([label, value]) => {
        let e = node("div"),
            v = node("div", fmt(value));
        v.className = "number";
        e.append(v, node("div", label));
        e.lastChild.className = "muted";
        return e;
    }));
}

function mechanics() {
    let rho = Number($("rho").value),
        eta = Number($("eta").value),
        eigen = [1 + rho, 1 - rho],
        stable = eta < 2 * eigen[1],
        ula = stable ? eigen.map(v => v / (1 - eta / (2 * v))) : null;
    $("rho-value").textContent = rho.toFixed(2);
    $("eta-value").textContent = eta.toFixed(3);
    numbers("ula-values", [
        ["Target large variance", eigen[0]],
        ["Target narrow variance", eigen[1]],
        ["ULA large variance", ula ? ula[0] : null],
        ["ULA narrow variance", ula ? ula[1] : null]
    ]);
    $("ula-scope").textContent = stable ? `Stable when η < ${fmt(2*eigen[1])}. ULA covariance eigenvalue λ/(1−η/(2λ)); inflation persists with arbitrarily long sampling. MALA adds the asymmetric Gaussian proposal correction to target the correct stationary law.` : `Unstable: η ≥ ${fmt(2*eigen[1])}. No finite stationary ULA covariance exists for this Gaussian. Smaller steps or a corrected/preconditioned sampler change the behavior.`;
    let svg = $("ellipses");
    svg.replaceChildren();
    let maximum = ula ? Math.sqrt(Math.max(...ula, ...eigen)) : Math.sqrt(eigen[0]),
        scale = 150 / maximum;

    function ellipse(values, color) {
        let path = document.createElementNS("http://www.w3.org/2000/svg", "path"),
            points = [];
        for (let i = 0; i <= 100; i++) {
            let t = i / 100 * 2 * Math.PI,
                u = Math.sqrt(values[0]) * Math.cos(t),
                v = Math.sqrt(values[1]) * Math.sin(t),
                x = (u + v) / Math.sqrt(2),
                y = (u - v) / Math.sqrt(2);
            points.push(`${i?"L":"M"}${320+scale*x},${220-scale*y}`);
        }
        path.setAttribute("d", points.join(" "));
        path.setAttribute("stroke", color);
        path.setAttribute("fill", "none");
        path.setAttribute("stroke-width", "3");
        svg.append(path);
    }
    ellipse(eigen, colors[0]);
    if (ula) ellipse(ula, colors[2]);
    let axis = document.createElementNS("http://www.w3.org/2000/svg", "text");
    axis.setAttribute("x", "30");
    axis.setAttribute("y", "420");
    axis.setAttribute("fill", "#587082");
    axis.textContent = `Equal x/y view: ±${(200/scale).toFixed(2)}`;
    svg.append(axis);
    let h = Number($("h").value),
        steps = Number($("steps").value),
        x = .8,
        p = .3,
        initial = .5 * (x * x + p * p),
        trajectory = [
            [x, p]
        ];
    for (let i = 0; i < steps; i++) {
        p -= h / 2 * x;
        x += h * p;
        p -= h / 2 * x;
        trajectory.push([x, p]);
    }
    let energy = .5 * (x * x + p * p),
        probability = Math.exp(Math.min(0, initial - energy));
    window.liveMechanics = {
        rho,
        eta,
        stable,
        targetEigen: eigen,
        ulaEigen: ula,
        h,
        steps,
        x,
        p,
        energy,
        acceptance: probability
    };
    $("h-value").textContent = h.toFixed(2);
    $("steps-value").textContent = steps;
    $("hamiltonian-values").textContent = `Final energy ${energy.toFixed(6)}; initial ${initial.toFixed(6)}; ΔH ${(energy-initial).toExponential(3)}; Metropolis acceptance probability ${probability.toFixed(6)}.`;
    const canvas = $("hamiltonian"),
        ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 640, 450);
    ctx.strokeStyle = "#dce7eb";
    ctx.beginPath();
    ctx.arc(320, 225, Math.sqrt(2 * initial) * 180, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = colors[0];
    ctx.lineWidth = 2;
    ctx.beginPath();
    trajectory.forEach(([a, b], i) => i ? ctx.lineTo(320 + 180 * a, 225 - 180 * b) : ctx.moveTo(320 + 180 * a, 225 - 180 * b));
    ctx.stroke();
    ctx.fillStyle = colors[2];
    ctx.beginPath();
    ctx.arc(320 + 180 * x, 225 - 180 * p, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "#587082";
    ctx.font = "13px system-ui";
    ctx.fillText("Position x (horizontal), momentum p (vertical); equal axes", 25, 425);
}

function candidates() {
    return evidence.rows.filter(r => r.group === group && r.task === $("task").value);
}

function selected() {
    return candidates().find(r => r.preset === $("preset").value);
}

function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    $("play").textContent = "Play retained chain states";
}

const descriptions = {
    gaussian: ["Correlated Gaussian", "A tilted 2D Gaussian: mean (2, −2), unit marginal variances and correlation 0.95. A correct sampler must recover both the ellipse and the two bell-shaped marginals.", "x₁", "x₂"],
    perturbed: ["Wavy Gaussian", "The original coursework target: a correlated Gaussian multiplied by exp(cos(x₁) + sin(x₂)). The waves reshape the density. Independent rejection draws supply the reference.", "x₁", "x₂"],
    ring: ["Eight-mode Gaussian mixture", "Eight Gaussian islands on a radius-three ring, with equal weights and component standard deviation 0.35. Sampling one island well is not the same as discovering all eight.", "x₁", "x₂"],
    banana: ["Banana distribution", "Draw u ∼ Normal(0, 4) and v ∼ Normal(0, 1); then x₁ = u and x₂ = v + 0.25(u² − 4). The target bends into a banana, which makes straight proposals inefficient.", "x₁", "x₂"],
    funnel: ["Neal's funnel, centered", "Draw y ∼ Normal(0, 9), then z | y ∼ Normal(0, exp(y)). The narrow neck and very wide tails make one fixed step size difficult. The axes are original-space y and z.", "y", "z"],
    funnel_noncentered: ["The same funnel, non-centered", "Sample independent y ∼ Normal(0, 9) and u ∼ Normal(0, 1), then display z = exp(y/2)u. This is the same original-space distribution as the centered funnel, sampled in easier coordinates.", "y", "z"],
    regression: ["Bayesian regression posterior", "Diabetes regression has eleven posterior coefficients including the intercept. This plot shows the exact 2D marginal of the first two coefficients, not a slice through the eleven-dimensional density.", "Intercept", "First feature coefficient"],
    logistic: ["Bayesian classification posterior", "Wisconsin Diagnostic classification has 31 posterior coefficients including the intercept. These are the first two sampled coefficients; no exact 2D posterior marginal is claimed. Predictive performance uses all coefficients.", "Intercept", "First feature coefficient"]
};
const explanations = {
    rwm: "Metropolis proposes a symmetric random step and accepts using the target-density ratio. Rejected proposals repeat the current state.",
    adaptive_rwm: "Adaptive Metropolis tunes its proposal scale during warmup, then freezes it for the retained samples shown here.",
    gibbs: "Gibbs samples exact conditional distributions: one Gaussian coordinate at a time, or an alternating mixture-label/position update for the ring. Exact conditionals do not guarantee fast movement between islands.",
    slice: "Slice sampling draws a horizontal density level, brackets a coordinate interval and samples inside the slice.",
    ula: "ULA takes a gradient step toward higher density plus Gaussian noise. There is no Metropolis correction: a stable chain can still sample a biased distribution.",
    mala: "MALA corrects the Langevin proposal using the forward and reverse Gaussian proposal densities. The correction targets the intended distribution.",
    hmc: "HMC draws momentum, follows a leapfrog trajectory and accepts or rejects the endpoint using the Hamiltonian energy change. The plotted path joins retained endpoints, not the internal leapfrog steps.",
    adaptive_hmc: "Adaptive HMC learns a step size and diagonal geometry during warmup. Retained draws use the frozen Hamiltonian transition with a Metropolis correction.",
    nuts: "NUTS adaptively builds Hamiltonian trajectories until they turn back. This uses Pyro's implementation; the cloud contains retained posterior draws.",
    parallel_tempering: "Parallel tempering runs eight temperatures and swaps neighboring replicas. Only the cold replica's samples estimate the target; the additional replicas have additional computation cost.",
    sgld: "SGLD uses an unbiased minibatch likelihood gradient and a decreasing Langevin step. These finite-budget samples are approximate; diagnostics alone cannot establish posterior accuracy.",
    sghmc: "The pedagogical SGHMC preset uses minibatch gradients, momentum and friction with a zero gradient-noise correction approximation. It is explicitly compared with full-gradient NUTS."
};

function contour(ctx, target, px, py) {
    if (!target.density) return;
    const grid = target.density,
        n = grid.length;
    const gx = j => target.bounds[0][0] + j / (n - 1) * (target.bounds[0][1] - target.bounds[0][0]);
    const gy = i => target.bounds[1][0] + i / (n - 1) * (target.bounds[1][1] - target.bounds[1][0]);
    ctx.strokeStyle = "#81939d";
    ctx.lineWidth = 1.3;
    ctx.globalAlpha = .65;
    for (const level of [.015, .08, .3, .7]) {
        ctx.beginPath();
        for (let i = 0; i < n - 1; i++)
            for (let j = 0; j < n - 1; j++) {
                const vertices = [
                    [gx(j), gy(i)],
                    [gx(j + 1), gy(i)],
                    [gx(j + 1), gy(i + 1)],
                    [gx(j), gy(i + 1)]
                ];
                const values = [grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]];
                const intersections = [];
                for (let k = 0; k < 4; k++) {
                    const l = (k + 1) % 4;
                    if ((values[k] >= level) === (values[l] >= level)) continue;
                    const t = (level - values[k]) / (values[l] - values[k]);
                    intersections.push(vertices[k].map((v, a) => v + t * (vertices[l][a] - v)));
                }
                for (let k = 0; k + 1 < intersections.length; k += 2) {
                    ctx.moveTo(px(intersections[k][0]), py(intersections[k][1]));
                    ctx.lineTo(px(intersections[k + 1][0]), py(intersections[k + 1][1]));
                }
            }
        ctx.stroke();
    }
    ctx.globalAlpha = 1;
}

function marginalPlot(row, indices, draw, bounds) {
    const ctx = $("marginal-canvas").getContext("2d"),
        labels = descriptions[row.task];
    ctx.clearRect(0, 0, 680, 320);
    const output = [];
    for (let axis = 0; axis < 2; axis++) {
        const [lo, hi] = bounds[axis], width = (hi - lo) / 40;
        const bars = indices.map(index => {
            const counts = Array(40).fill(0);
            for (const point of row.points[index].slice(0, draw)) {
                if (point[axis] >= lo && point[axis] <= hi) counts[Math.min(39, Math.floor((point[axis] - lo) / width))]++;
            }
            return {
                chain: index,
                counts,
                density: counts.map(n => n / (draw * width))
            };
        });
        const reference = evidence.targets[row.task].marginal_reference?.[axis];
        const peak = Math.max(1e-12, ...bars.flatMap(b => b.density), ...(reference ? reference.density : []));
        const left = 55 + axis * 335,
            bottom = 250,
            height = 190,
            w = 260;
        const px = x => left + (x - lo) / (hi - lo) * w,
            py = y => bottom - height * y / peak;
        ctx.fillStyle = "#587082";
        ctx.font = "13px system-ui";
        ctx.fillText(labels[axis + 2] + " marginal density", left, 30);
        ctx.fillText(lo.toPrecision(3), left, 275);
        ctx.fillText(hi.toPrecision(3), left + 220, 275);
        ctx.fillText("Density", left, 50);
        ctx.strokeStyle = "#dce7eb";
        ctx.beginPath();
        ctx.moveTo(left, 55);
        ctx.lineTo(left, bottom);
        ctx.lineTo(left + w, bottom);
        ctx.stroke();
        ctx.save();
        ctx.beginPath();
        ctx.rect(left, 55, w, height + 5);
        ctx.clip();
        for (const bar of bars) {
            ctx.fillStyle = colors[bar.chain];
            ctx.globalAlpha = indices.length === 1 ? .55 : .22;
            bar.density.forEach((v, k) => ctx.fillRect(px(lo + k * width), py(v), w / 40, bottom - py(v)));
        }
        ctx.globalAlpha = 1;
        if (reference) {
            ctx.strokeStyle = "#183246";
            ctx.lineWidth = 2;
            ctx.beginPath();
            reference.density.forEach((v, k) => {
                const x = (reference.edges[k] + reference.edges[k + 1]) / 2;
                k ? ctx.lineTo(px(x), py(v)) : ctx.moveTo(px(x), py(v));
            });
            ctx.stroke();
        }
        ctx.restore();
        output.push({
            bounds: [lo, hi],
            bin_width: width,
            chains: bars
        });
    }
    $("marginal-caption").textContent = evidence.targets[row.task].marginal_reference ? "Each chain has its own colored histogram. Black reference: 100,000 independent exact draws, fixed seed 800000; the curve spans the target-focus window. Repeated samples count as repeated draws. A matching pair of marginals does not prove joint mixing." : "Sampled coefficient marginals only. No exact reference curve is available for this 31-dimensional posterior. Use the predictive results and full-chain diagnostics alongside these projections.";
    return output;
}

function chain() {
    const row = selected();
    if (row.kind !== "sampling") return;
    const draw = Number($("draw").value),
        selectedChain = $("chain").value;
    const indices = selectedChain === "all" ? [0, 1, 2, 3] : [Number(selectedChain)];
    const target = evidence.targets[row.task];
    if (!row.fullBounds) {
        const points = row.points.flat();
        row.fullBounds = [0, 1].map(axis => {
            const values = points.map(p => p[axis]),
                lo = Math.min(...values),
                hi = Math.max(...values);
            const pad = Math.max(hi - lo, 1e-6) * .05;
            return [lo - pad, hi + pad];
        });
    }
    const bounds = $("plot-window").value === "focus" ? target.bounds : row.fullBounds;
    const ctx = $("chain-canvas").getContext("2d");
    const px = x => 75 + (x - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * 540;
    const py = y => 450 - (y - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) * 395;
    ctx.clearRect(0, 0, 680, 540);
    ctx.strokeStyle = "#dce7eb";
    ctx.beginPath();
    ctx.rect(75, 55, 540, 395);
    ctx.stroke();
    ctx.save();
    ctx.beginPath();
    ctx.rect(75, 55, 540, 395);
    ctx.clip();
    contour(ctx, target, px, py);
    let outside = 0;
    const clouds = [];
    for (const i of indices) {
        const points = row.points[i].slice(0, draw);
        outside += points.filter(p => p.some((v, a) => v < bounds[a][0] || v > bounds[a][1])).length;
        ctx.strokeStyle = ctx.fillStyle = colors[i];
        ctx.globalAlpha = $("sample-view").value === "paths" ? .35 : .4;
        ctx.lineWidth = .9;
        ctx.beginPath();
        if ($("sample-view").value === "paths") {
            points.forEach((p, j) => j ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1])));
            ctx.stroke();
        } else {
            points.forEach(p => {
                ctx.moveTo(px(p[0]) + 1.7, py(p[1]));
                ctx.arc(px(p[0]), py(p[1]), 1.7, 0, Math.PI * 2);
            });
            ctx.fill();
        }
        ctx.globalAlpha = 1;
        ctx.beginPath();
        const last = points[points.length - 1];
        ctx.arc(px(last[0]), py(last[1]), 4, 0, Math.PI * 2);
        ctx.fill();
        clouds.push({
            chain: i,
            draws: points.length
        });
    }
    ctx.restore();
    ctx.fillStyle = "#587082";
    ctx.font = "12px system-ui";
    for (let k = 0; k <= 4; k++) {
        const x = bounds[0][0] + k / 4 * (bounds[0][1] - bounds[0][0]);
        const y = bounds[1][0] + k / 4 * (bounds[1][1] - bounds[1][0]);
        ctx.fillText(x.toPrecision(3), px(x) - 14, 475);
        ctx.fillText(y.toPrecision(3), 12, py(y) + 4);
    }
    const description = descriptions[row.task];
    ctx.fillText(description[2] + " →", 275, 510);
    ctx.fillText(description[3] + " ↑", 75, 35);
    $("distribution-title").textContent = description[0];
    $("target-explanation").textContent = description[1];
    $("algorithm-explanation").textContent = explanations[row.metrics.method];
    $("draw-value").textContent = draw;
    $("chain-caption").textContent = `${indices.length * draw} retained samples accumulated (${draw} per selected chain). ${outside} samples lie outside the current plot window. Axes have independent, labeled scales. ${target.scope} Warmup is excluded; this is replay of actual recorded sampling.`;
    const histograms = marginalPlot(row, indices, draw, bounds);
    window.sampleViewEvidence = {
        name: row.name,
        draw,
        bounds,
        outside,
        displayed: indices.length * draw,
        clouds,
        histograms,
        view: $("sample-view").value
    };
    window.replayEvidence = {
        name: row.name,
        draw,
        points: row.points.map(c => c[Math.min(draw - 1, c.length - 1)])
    };
}

function render() {
    stop();
    const row = selected(),
        m = row.metrics;
    $("title").textContent = row.preset;
    $("kind").textContent = row.kind === "sampling" ? "Four independent measured chains" : "Owned normalizable neural energy";
    $("replay-controls").classList.toggle("hidden", row.kind !== "sampling");
    $("figure").src = row.image;
    $("figure").alt = row.name + " diagnostics";
    $("raw").textContent = JSON.stringify(m, null, 2);
    $("predictive-results").replaceChildren();
    if (m.predictive) {
        $("predictive-results").append(node("h2", "What can these posterior samples predict?"));
        const p = m.predictive.test;
        $("predictive-results").append(node("p", row.task === "regression" ? `Held-out Diabetes RMSE (standardized target): ${fmt(p.rmse_standardized)}. Predictive interval coverage: ${fmt(100*p.predictive_interval_95_coverage)}%.` : `Held-out Wisconsin Diagnostic AUROC: ${fmt(p.roc_auc)}. Log loss: ${fmt(p.log_loss)}. Brier score: ${fmt(p.brier_score)}.`));
    }
    $("source").href = row.source;
    $("numeric").href = row.numeric;
    $("scope").textContent = m.scope + (row.kind === "energy" ? " " + m.integrability : "");
    if (row.kind === "sampling") {
        let d = m.diagnostics;
        numbers("numbers", [
            ["Maximum rank Rhat", extrema(d.rank_rhat, "max")],
            ["Minimum bulk ESS", extrema(d.bulk_ess, "min")],
            ["Minimum tail ESS", extrema(d.tail_ess, "min")],
            ["Divergences", d.divergences],
            ["Diagnostic screen", d.convergence_flag ? "Passed" : "Failed"],
            ["Seconds incl. warmup", d.total_sampling_and_warmup_seconds]
        ]);
        $("draw").max = row.points[0].length;
        $("draw").value = row.points[0].length;
        chain();
    } else numbers("numbers", [
        ["Selected update", m.selected_step],
        ["Validation NLL", m.validation_nll],
        ["Held-out NLL", m.test_nll],
        ["Integration refinement change", m.partition_max_change],
        ["Negative-phase gradients", m.negative_gradient_evaluations],
        ["Fitting seconds", m.fitting_seconds]
    ]);
    let table = node("table"),
        head = node("tr");
    (row.kind === "sampling" ? ["Algorithm / preset", "Max Rhat", "Min bulk ESS", "Min tail ESS", "Divergences", "Screen"] : ["Negative phase / seed", "Test NLL", "Selected update", "Seconds"]).forEach(v => head.append(node("th", v)));
    table.append(head);
    candidates().forEach(r => {
        let tr = node("tr"),
            q = r.metrics,
            d = q.diagnostics,
            values = row.kind === "sampling" ? [r.preset, extrema(d.rank_rhat, "max"), extrema(d.bulk_ess, "min"), extrema(d.tail_ess, "min"), d.divergences, d.convergence_flag ? "Passed" : "Failed"] : [r.preset, q.test_nll, q.selected_step, q.fitting_seconds];
        values.forEach(v => tr.append(node("td", fmt(v))));
        table.append(tr);
    });
    $("comparison").replaceChildren(table);
}

function presets() {
    options("preset", candidates().map(r => r.preset));
    if (group === "Chain diagnostics" && $("task").value === "gaussian") $("preset").value = "gibbs · dispersed · preset 1";
    render();
}

function select(value) {
    stop();
    group = value;
    document.querySelectorAll("nav button").forEach(b => b.setAttribute("aria-selected", String(b.textContent === group)));
    $("mechanics").classList.toggle("hidden", group !== groups[0]);
    $("measured").classList.toggle("hidden", group === groups[0]);
    if (group === groups[0]) {
        mechanics();
        return;
    }
    options("task", [...new Set(evidence.rows.filter(r => r.group === group).map(r => r.task))]);
    presets();
}
groups.forEach(g => {
    let b = node("button", g);
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => select(g));
    $("concepts").append(b);
});
$("task").addEventListener("change", presets);
$("preset").addEventListener("change", render);
$("draw").addEventListener("input", chain);
$("chain").addEventListener("change", chain);
["sample-view", "plot-window"].forEach(id => $(id).addEventListener("change", chain));
["rho", "eta", "h", "steps"].forEach(id => $(id).addEventListener("input", mechanics));
$("play").addEventListener("click", () => {
    if (timer) {
        stop();
        return;
    }
    if (Number($("draw").value) >= Number($("draw").max)) {
        $("draw").value = 1;
        chain();
    }
    $("play").textContent = "Pause";
    timer = setInterval(() => {
        let v = Number($("draw").value) + 10;
        if (v > Number($("draw").max)) {
            stop();
            return;
        }
        $("draw").value = v;
        chain();
    }, 70);
});
$("chain-legend").replaceChildren(...colors.map((color, i) => {
    let e = node("span", `Chain ${i}`);
    e.style.color = color;
    return e;
}));
select("Chain diagnostics");
