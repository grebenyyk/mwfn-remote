"""Minimal TOML-subset parser (sections, scalars, arrays of numbers) with
tomli/tomllib used automatically when available. Enough for mwfn.toml."""
from __future__ import annotations

try:                                    # python >= 3.11
    import tomllib as _toml
except ImportError:
    try:
        import tomli as _toml
    except ImportError:
        _toml = None


def _parse_scalar(s: str):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        return float(s)


def _mini_toml(text: str) -> dict:
    data, section = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            items = [v for v in val[1:-1].split(",") if v.strip()]
            parsed = [_parse_scalar(v) for v in items]
        else:
            parsed = _parse_scalar(val)
        (data[section] if section else data)[key] = parsed
    return data


def load_config(path: str) -> dict:
    text = open(path).read()
    if _toml is not None:
        return _toml.loads(text)
    return _mini_toml(text)
