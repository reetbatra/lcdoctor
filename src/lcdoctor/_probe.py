"""Runs INSIDE the interpreter being checked, as a standalone script.

It must only use the standard library and stay compatible with Python 3.8
syntax, because the environment it runs in is the user's, not lcdoctor's.

stdin:  {"targets": [{"module": str, "name": str | null}, ...],
         "family_prefixes": [str, ...]}
stdout: one JSON document (see ``main``).
"""

import importlib
import inspect
import json
import re
import sys
import sysconfig
import warnings
from typing import Any, Dict

# LangChain also signals moves with plain UserWarnings, e.g. "Importing X from
# langchain root module is no longer supported. Please use Y instead."
MOVE_PHRASES = (
    "deprecat",
    "no longer supported",
    "has been moved",
    "will be removed",
    "please import from",
)
SINCE_RE = re.compile(r"^\s*v?(\d+\.\d+(?:\.\d+)?(?:[a-z]+\d*)?)\b")
REMOVAL_RE = re.compile(
    r"removed\s+(?:in|until)\s+([A-Za-z0-9_.\-]+)\s*==\s*([0-9][A-Za-z0-9_.]*[0-9A-Za-z])"
)
# langchain-core >= 1.0 renders: !!! deprecated "0.1.17 Use X instead. It will be removed in pkg==2.0.0."
ADMONITION_RE = re.compile(r'!!!\s*deprecated\s*"(.*?)"\s*(?:\n|$)', re.DOTALL)
# langchain-core < 1.0 renders: .. deprecated:: 0.1.17 Use X instead. (continues until a blank line)
SPHINX_RE = re.compile(r"\.\.\s*deprecated::\s*(.*?)(?:\n\s*\n|$)", re.DOTALL)


def _clean(text):
    text = text.replace("``", "`")
    text = re.sub(r":(?:class|meth|func|attr|mod|obj):`~?([^`]+)`", r"`\1`", text)
    text = re.sub(r"`+", "`", text)
    return " ".join(text.split())


def parse_deprecation_doc(doc):
    """Extract since/removal/message from a LangChain-rendered docstring."""
    if not doc:
        return None
    match = ADMONITION_RE.search(doc) or SPHINX_RE.search(doc)
    if not match:
        return None
    body = _clean(match.group(1))
    since = None
    since_match = SINCE_RE.match(body)
    if since_match:
        since = since_match.group(1)
        body = body[since_match.end() :].strip()
    removal = None
    removal_match = REMOVAL_RE.search(body)
    if removal_match:
        removal = {"package": removal_match.group(1), "version": removal_match.group(2).rstrip(".")}
    return {"since": since, "removal": removal, "message": body}


def describe_deprecation(obj):
    info = None
    try:
        doc = inspect.getdoc(obj)
    except Exception:
        doc = None
    parsed = parse_deprecation_doc(doc)
    marker = getattr(obj, "__deprecated__", None)
    if isinstance(marker, str) and marker:
        info = {
            "since": None,
            "removal": None,
            "message": _clean(marker),
            "source": "__deprecated__",
        }
    if parsed:
        merged = {"since": parsed["since"], "removal": parsed["removal"], "source": "docstring"}
        # The docstring message is the fuller one; keep the PEP 702 text if it says more.
        message = parsed["message"]
        if info and len(info["message"]) > len(message):
            message = info["message"]
        merged["message"] = message
        if info:
            merged["source"] = "__deprecated__+docstring"
        info = merged
    return info


def _kind(obj):
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "value"


def _deprecation_warnings(caught):
    out = []
    for w in caught:
        cat = w.category
        text = str(w.message)
        lowered = text.lower()
        if issubclass(cat, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)) or any(
            phrase in lowered for phrase in MOVE_PHRASES
        ):
            msg = _clean(text)
            if msg not in out:
                out.append(msg)
    return out


def probe_one(module_name, attr_name, module_cache):
    result = {
        "module": module_name,
        "name": attr_name,
        "status": "ok",
        "kind": None,
        "defined_in": None,
        "error": None,
        "missing_module": None,
        "deprecation": None,
        "warnings": [],
    }
    if module_name not in module_cache:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                module_cache[module_name] = (importlib.import_module(module_name), None)
            except ModuleNotFoundError as exc:
                module_cache[module_name] = (None, ("missing_module", exc.name, str(exc)))
            except BaseException as exc:  # a dependency failing to import is a finding, not a crash
                module_cache[module_name] = (
                    None,
                    ("import_error", None, "%s: %s" % (type(exc).__name__, exc)),
                )
        module_cache[module_name + "\0warnings"] = _deprecation_warnings(caught)
    module, failure = module_cache[module_name]
    result["warnings"] = list(module_cache[module_name + "\0warnings"])
    if failure:
        status, missing, message = failure
        top = module_name.split(".", 1)[0]
        if status == "missing_module":
            # Which module was missing tells us *why*: the package, a submodule, or a dependency.
            if missing == top:
                status = "not_installed"
            elif missing and not (missing == module_name or module_name.startswith(missing + ".")):
                status = "import_error"
        result.update(status=status, missing_module=missing, error=message)
        return result
    if attr_name is None:
        result["kind"] = "module"
        result["deprecation"] = describe_deprecation(module)
        return result
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = getattr(module, attr_name)
        except AttributeError as exc:
            # `from pkg import sub` also works when `sub` is a submodule nobody imported yet.
            obj = None
            result.update(status="missing_name", error=str(exc))
            try:
                obj = importlib.import_module(module_name + "." + attr_name)
                result.update(status="ok", error=None)
            except ModuleNotFoundError as sub_exc:
                if sub_exc.name != module_name + "." + attr_name:
                    result.update(
                        status="import_error",
                        missing_module=sub_exc.name,
                        error="%s: %s" % (type(sub_exc).__name__, sub_exc),
                    )
            except BaseException as sub_exc:
                result.update(
                    status="import_error", error="%s: %s" % (type(sub_exc).__name__, sub_exc)
                )
        except BaseException as exc:
            obj = None
            result.update(status="import_error", error="%s: %s" % (type(exc).__name__, exc))
    for w in _deprecation_warnings(caught):
        if w not in result["warnings"]:
            result["warnings"].append(w)
    if result["status"] == "ok":
        # getattr can also succeed on a submodule that was never imported explicitly.
        result["kind"] = _kind(obj)
        result["deprecation"] = describe_deprecation(obj)
        if result["kind"] in ("class", "function"):
            defined = getattr(obj, "__module__", None)
            result["defined_in"] = defined if isinstance(defined, str) else None
    return result


def family_distributions(prefixes):
    try:
        from importlib import metadata
    except ImportError:  # Python < 3.8
        return {}
    out = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        norm = name.lower().replace("_", "-")
        if any(norm == p or norm.startswith(p + "-") for p in prefixes) or norm == "langsmith":
            out[norm] = dist.version
    return dict(sorted(out.items()))


def module_distributions(tops):
    try:
        from importlib import metadata

        mapping = metadata.packages_distributions()
    except Exception:
        return {}
    return {
        top: sorted({d.lower().replace("_", "-") for d in mapping.get(top, [])}) for top in tops
    }


def main():
    payload = json.load(sys.stdin)
    prefixes = payload.get("family_prefixes", ["langchain", "langgraph"])
    cache: Dict[str, Any] = {}
    results = [probe_one(t["module"], t.get("name"), cache) for t in payload["targets"]]
    tops = sorted({t["module"].split(".", 1)[0] for t in payload["targets"]})
    paths = sysconfig.get_paths()
    site = [p for p in {paths.get("purelib"), paths.get("platlib")} if p]
    out = {
        "python": "%d.%d.%d" % sys.version_info[:3],
        "executable": sys.executable,
        "site_packages": sorted(site),
        "distributions": family_distributions(prefixes),
        "module_distributions": module_distributions(tops),
        "results": results,
    }
    # Imported libraries may print to stdout; the JSON document is fenced so it can be found.
    sys.stdout.write("\n@@LCDOCTOR@@" + json.dumps(out) + "@@LCDOCTOR@@\n")


if __name__ == "__main__":
    main()
