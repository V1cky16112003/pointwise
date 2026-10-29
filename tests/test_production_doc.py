"""docs/production.md's env var table must match the code, both ways, with the same defaults."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "production.md"
WANTED = re.compile(r"POINTWISE_\w+|ADAPTER_DIR|HF_HOME")
ENV_FUNCS = {"env_int", "env_float", "env_flag", "_env_int", "_env_float", "_env_flag"}
UNSET = object()


def _module_constants() -> dict[str, object]:
    """Module-level NAME = value (and NAME, NAME = a, b) across src/, so a default given as a constant resolves."""
    out = {}
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
                pairs = (zip(target.elts, value.elts) if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
                         else [(target, value)])
                for t, v in pairs:
                    if isinstance(t, ast.Name):
                        try:
                            out[t.id] = eval(compile(ast.Expression(v), "", "eval"), {})  # constants like 1 << 20
                        except Exception:  # noqa: BLE001  not a constant
                            pass
    return out


def code_env_defaults() -> dict[str, object]:
    """Every wanted env var name src/ reads through os.environ.get or env_*, with its default (UNSET if none)."""
    consts, found = _module_constants(), {}
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            func = node.func
            is_get = isinstance(func, ast.Attribute) and func.attr == "get" and ast.unparse(func.value) == "os.environ"
            is_env = isinstance(func, ast.Name) and func.id in ENV_FUNCS
            name = node.args[0].value
            if not (is_get or is_env) or not isinstance(name, str) or not WANTED.fullmatch(name):
                continue
            default = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "default"), None)
            if default is None:
                value = False if is_env and func.id.endswith("flag") else UNSET
            elif isinstance(default, ast.Name):
                value = consts[default.id]
            else:
                value = ast.literal_eval(default)
            found[name] = UNSET if value == "" else value
    found["HF_HOME"] = re.search(r"HF_HOME=(\S+)", (ROOT / "Dockerfile").read_text()).group(1)  # read by the HF libraries
    return found


def doc_env_defaults(text: str) -> dict[str, str]:
    """Rows of the configuration table: `NAME` | default | ..."""
    section = text.split("## 3. Configuration", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `([A-Z_]+)` \| ([^|]+) \|", section, re.M)
    return {name: cell.strip().strip("`") for name, cell in rows}


def same(doc: str, code: object) -> bool:
    if code is UNSET:
        return doc == "unset"
    if isinstance(code, bool):
        return doc == str(int(code))
    if isinstance(code, (int, float)):
        return float(doc) == float(code)
    return doc == code


def compare(doc: dict[str, str], code: dict[str, object]) -> list[str]:
    problems = [f"{n} is read by src/ but has no row" for n in sorted(code.keys() - doc.keys())]
    problems += [f"{n} has a row but src/ does not read it" for n in sorted(doc.keys() - code.keys())]
    problems += [f"{n}: doc default {doc[n]!r} != code {code[n]!r}" for n in sorted(doc.keys() & code.keys())
                 if not same(doc[n], code[n])]
    return problems


def test_env_table_matches_the_code():
    doc = doc_env_defaults(DOC.read_text())
    assert len(doc) >= 15  # the table parsed
    assert compare(doc, code_env_defaults()) == []


def test_compare_catches_a_missing_row_and_a_wrong_default():
    code = {"POINTWISE_A": 16, "POINTWISE_B": UNSET}
    assert compare({"POINTWISE_A": "16"}, code) == ["POINTWISE_B is read by src/ but has no row"]
    assert compare({"POINTWISE_A": "17", "POINTWISE_B": "unset"}, code) == ["POINTWISE_A: doc default '17' != code 16"]
