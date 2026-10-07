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

function chain() {
    const row = selected();
    if (row.kind !== "sampling") return;
    const draw = Number($("draw").value),
        selectedChain = $("chain").value,
        points = row.points.flat(),
        mins = [Math.min(...points.map(p => p[0])), Math.min(...points.map(p => p[1]))],
        maxs = [Math.max(...points.map(p => p[0])), Math.max(...points.map(p => p[1]))],
        center = mins.map((v, i) => (v + maxs[i]) / 2),
        span = Math.max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-6) * 1.1,
        scale = 440 / span,
        ctx = $("chain-canvas").getContext("2d"),
        px = x => 340 + (x - center[0]) * scale,
        py = y => 270 - (y - center[1]) * scale;
    ctx.clearRect(0, 0, 680, 540);
    row.points.forEach((values, i) => {
        if (selectedChain !== "all" && Number(selectedChain) !== i) return;
        ctx.strokeStyle = colors[i];
        ctx.globalAlpha = .3;
        ctx.lineWidth = .8;
        ctx.beginPath();
        values.slice(0, draw).forEach((p, j) => j ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1])));
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = colors[i];
        let p = values[Math.min(draw - 1, values.length - 1)];
        ctx.beginPath();
        ctx.arc(px(p[0]), py(p[1]), 4, 0, 2 * Math.PI);
        ctx.fill();
    });
    ctx.fillStyle = "#587082";
    ctx.font = "12px system-ui";
    ctx.fillText(`x: ${(center[0]-span/2).toFixed(1)}…${(center[0]+span/2).toFixed(1)}; y: ${(center[1]-span/2).toFixed(1)}…${(center[1]+span/2).toFixed(1)}; equal axes, full extent`, 20, 525);
    $("draw-value").textContent = draw;
    $("chain-caption").textContent = "Replay of unthinned retained states after warmup. Full-draw axes stay fixed during playback. Large excursions can compress the view; they are retained. Non-centered funnel coordinates are transformed to the original funnel before visualization and diagnostics.";
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
        $("draw").value = Math.min(500, row.points[0].length);
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
["rho", "eta", "h", "steps"].forEach(id => $(id).addEventListener("input", mechanics));
$("play").addEventListener("click", () => {
    if (timer) {
        stop();
        return;
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
select(groups[0]);
