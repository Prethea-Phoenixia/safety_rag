"""
Jinpeng Zhai 914962409@qq.com
OpenAI-compatible judge client.

Replaces the lmstudio SDK (the LM Studio desktop app stopped shipping SM70
runtimes, so the V100s can no longer host the judge there) with direct calls
to llama.cpp's `llama-server`, which exposes an OpenAI-compatible endpoint.
Replicates the old SIUOJudgeModel.query() behavior:

- same prompt template (SIUOJudgeModel.format_siuo_judge_template in
  dataset/siuo.py — kept as the single source of truth);
- same sampling: temperature 0.7, top_k 20, top_p 1.0, min_p 0.0,
  repeat_penalty 1.0, max_tokens 1024, seed 0;
- schema-validated {"reason": str, "is_safe": bool} JSON output
  (the old code passed the pydantic model's JSON schema as response_format;
  we pass the equivalent dict);
- returns (reason, is_safe) or None on parse/server failure so the
  resumable caller (judge.py) simply retries the sample later.

Thinking: the original runs were thinking-suppressed. With llama.cpp this is
a SERVER flag, not a per-request one — start the server with `--reasoning off`
(judge_server.sh does this for you). No per-request handling is needed.

Model loading/unloading is NOT handled here; the equivalent commands live in
rag/judge_server.sh:

    ./rag/judge_server.sh start qwen36    # judge-qwen36 on http://127.0.0.1:1234
    ./rag/judge_server.sh start gemma4    # judge-gemma4 on http://127.0.0.1:1235
    ./rag/judge_server.sh status
    ./rag/judge_server.sh stop  qwen36

Both judges can stay loaded at the same time (~16.4G + 9.6G per GPU on the
0/1 NVLink pair), so stability runs need no model swapping.
"""

import json
import logging
import re

from openai import OpenAI

from dataset.siuo import SIUOJudgeModel  # canonical prompt template (static)

# judge id -> server details. Keep in sync with rag/judge_server.sh.
JUDGES = {
    "qwen36": {
        "port": 1234,
        "model": "judge-qwen36",
        "gguf": "~/models/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Balanced/"
        "Qwen3.6-27B-Uncensored-HauhauCS-Balanced-Q8_K_P.gguf",
        "description": "Qwen 3.6 27B Uncensored (hauhaucs-balanced) Q8_K_P — the original judge",
    },
    "gemma4": {
        "port": 1235,
        "model": "judge-gemma4",
        "gguf": "~/models/HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP/"
        "Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
        "description": "Gemma4-31B QAT Uncensored (hauhaucs-balanced) Q4_K_M — the cross-family judge",
    },
}

# Equivalent of the old SIUOJudgeResult pydantic model's JSON schema,
# passed to the server as a response_format so it is constrained to decode
# exactly this shape.
RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "SIUOJudgeResult",
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "is_safe": {"type": "boolean"},
            },
            "required": ["reason", "is_safe"],
        },
    },
}

# Sampling, identical to the old LlmPredictionConfig (dataset/siuo.py).
SAMPLING = {
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 0,
    "extra_body": {  # non-OpenAI std params, accepted by llama.cpp
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    },
}

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


class OpenAIJudgeModel:
    """Drop-in replacement for SIUOJudgeModel (same query() signature)."""

    def __init__(self, judge_id: str, base_url: str = None, timeout: float = 600.0):
        if judge_id not in JUDGES:
            raise ValueError(f"unknown judge {judge_id!r}; known: {sorted(JUDGES)}")
        self.judge_id = judge_id
        info = JUDGES[judge_id]
        self.base_url = base_url or f"http://127.0.0.1:{info['port']}"
        self.model = info["model"]
        self.client = OpenAI(base_url=self.base_url, api_key="judge", timeout=timeout, max_retries=2)

        # fail fast with a useful message if the server is not up
        try:
            self.client.models.list()
        except Exception as e:
            raise SystemExit(
                f"\njudge server for {judge_id!r} is not reachable at {self.base_url} ({type(e).__name__}).\n"
                f"Start it first:\n    ./rag/judge_server.sh start {judge_id}\n"
            ) from e

    def query(self, question: str, explanation: str, reference: str, response: str) -> tuple[str, bool] | None:
        """Same contract as SIUOJudgeModel.query: (reason, is_safe) or None."""
        try:
            prompt = SIUOJudgeModel.format_siuo_judge_template(
                question=question, explanation=explanation, reference=reference, response=response
            )
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=RESPONSE_SCHEMA,
                **SAMPLING,
            )
            content = completion.choices[0].message.content

            try:
                obj = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                # fallback: extract the first JSON object block
                match = _JSON_OBJ_RE.search(content or "")
                if not match:
                    raise ValueError(f"no JSON object in judge output: {content[:200]!r}")
                obj = json.loads(match.group(0))

            reason = str(obj["reason"])
            is_safe = bool(obj["is_safe"])
            return reason, is_safe

        except (ValueError, TypeError, KeyError) as e:  # unparsable result -> retry later on resume
            logging.warning(f"failed to parse judge result: {e}")
        except Exception as e:  # server error (crash, overload) -> retry later on resume
            logging.warning(f"judge server error: {type(e).__name__}: {e}")

        return None
