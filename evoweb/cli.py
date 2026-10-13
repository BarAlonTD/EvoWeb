"""`evoweb run`: run the benchmark on a model, judged by another, via harness/run_local.sh.

A model (or judge) is a name plus where it is served:
  --model-port 8000 [--model-host H]   a vLLM / OpenAI-compatible server at http://H:8000/v1
  --model-url https://<host>/v1        any other OpenAI-compatible API, key in MODEL_API_KEY
                                       (JUDGE_API_KEY for the judge)
  neither                              an API model, `provider/model`, keyed by the provider's
                                       usual variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from evoweb.paths import ROOT

# provider prefix -> (harness PROVIDER, base URL, key variable)
APIS = {
    "anthropic": ("Anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "openai": ("OpenAILike", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def endpoint(name: str, host: str, port: int | None, url: str | None, key_var: str) -> tuple:
    """(provider, base URL, model id without provider prefix, key) for one role."""
    if port or url:
        # The benchmark runs in Docker: the host's localhost is host.docker.internal there.
        if host in ("localhost", "127.0.0.1"):
            host = "host.docker.internal"
        return (
            "OpenAILike",
            url or f"http://{host}:{port}/v1",
            name,
            os.environ.get(key_var, "local"),
        )
    prefix, _, model = name.partition("/")
    if prefix not in APIS or not model:
        sys.exit(f"{name}: give a port or URL, or a model prefixed with one of {', '.join(APIS)}")
    provider, base, var = APIS[prefix]
    if not os.environ.get(var):
        sys.exit(f"{name}: {var} is not set")
    return provider, base, model, os.environ[var]


def run(a: argparse.Namespace) -> int:
    provider, url, model, key = endpoint(
        a.model, a.model_host, a.model_port, a.model_url, "MODEL_API_KEY"
    )
    env = {
        "PROVIDER": provider,
        "API_KEY": key,
        # The agent takes Anthropic models as anthropic/<id> and everything else as openai/<id>.
        "MODEL": f"anthropic/{model}" if provider == "Anthropic" else f"openai/{model}",
    }
    if provider == "Anthropic":
        os.environ.pop("API_URL", None)
    else:
        # OpenRouter's agent base has no /v1 (the capture proxy adds it).
        env["API_URL"] = url.removesuffix("/v1") if provider == "OpenRouter" else url
    _, env["EVAL_API_URL"], env["EVAL_MODEL"], env["EVAL_API_KEY"] = endpoint(
        a.judge, a.judge_host, a.judge_port, a.judge_url, "JUDGE_API_KEY"
    )
    os.environ.update(env)
    return subprocess.call(
        [str(ROOT / "harness" / "run_local.sh"), a.name, a.apps, str(a.parallel)]
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="evoweb")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the benchmark; results go to runs/<name>/")
    r.add_argument("--name", required=True, help="run name; results go to runs/<name>/")
    for role, default in (("model", None), ("judge", "anthropic/claude-sonnet-4-6")):
        r.add_argument(
            f"--{role}",
            required=default is None,
            default=default,
            help="served name, or provider/model for an API",
        )
        r.add_argument(f"--{role}-port", type=int, help="port of a vLLM / OpenAI-compatible server")
        r.add_argument(f"--{role}-host", default="localhost", help="host of that server")
        r.add_argument(f"--{role}-url", help="full base URL instead of host + port")
    r.add_argument("--apps", default="1-32", help="subset, e.g. 1,4,6")
    r.add_argument("--parallel", type=int, default=8, help="apps run at once")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
