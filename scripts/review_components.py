"""Inventory the public component set and validate parseability, English and links."""

import ast
import json
import re
import subprocess
from pathlib import Path

import nbformat

from sampling_lab.artifacts import sha256, write_json


def main():
    components = []
    errors = []
    links = 0
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    files = sorted(set(Path(x.decode()) for x in result.stdout.split(b"\0") if x))
    excluded = {"legacy/originals", ".local", "dist", "build"}
    for path in files:
        if not path.is_file() or any(str(path).startswith(x + "/") for x in excluded):
            continue
        if path.stat().st_size > 20 * 1024**2:
            errors.append("Oversized public file: " + str(path))
        if path.suffix not in [".py", ".md", ".json", ".toml", ".yml", ".cff", ".ipynb"]:
            continue
        if path.name in ["component_review.json", "release_artifacts.json"]:
            continue
        text = path.read_text()
        component = dict(path=str(path), sha256=sha256(path), kind=path.suffix)
        if path.suffix == ".py":
            tree = ast.parse(text)
            component["functions"] = [
                n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            ]
        elif path.suffix == ".json":
            json.loads(text)
        elif path.suffix == ".ipynb":
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            text = "\n".join(c.source for c in nb.cells)
            component["code_cells"] = sum(c.cell_type == "code" for c in nb.cells)
        if re.search("[\u0600-\u06ff]", text):
            errors.append("Non-English script: " + str(path))
        if path.suffix in [".md", ".ipynb"]:
            prose = (
                text
                if path.suffix == ".md"
                else "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
            )
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", prose):
                if "://" in target or target.startswith("#"):
                    continue
                target = target.split("#")[0]
                links += 1
                if (
                    not (path.parent / target).exists()
                    and (path.parent / target).resolve()
                    != Path("docs/component_review.json").resolve()
                ):
                    errors.append(str(path) + " -> " + target)
        components.append(component)
    assert not errors, errors
    write_json(
        "docs/component_review.json",
        dict(
            status="passed",
            components=components,
            local_links=links,
            scope="Public inventory, Python AST/JSON/notebook syntax, English-script and local-link checks. Scientific manual review is documented separately; generated evidence is independently recomputed by validate_release.py.",
        ),
    )
    print("Reviewed", len(components), "public components;", links, "local links")


if __name__ == "__main__":
    main()
