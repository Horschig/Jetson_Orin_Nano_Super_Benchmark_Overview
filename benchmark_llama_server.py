#!/usr/bin/env python3
"""Production benchmark runner for llama.cpp llama-server.

This script benchmarks one or more GGUF models through llama-server only.
It measures load time, TTFT (client-side), prompt processing throughput (PP),
and token generation throughput (TG) over repeated runs.

Outputs:
- Per-run CSV
- Summary CSV
- JSON summary
- Per-run generated text files
- Per-run raw log slices
- Console summary table
- README benchmark section update
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError as exc:
    raise SystemExit("requests is required: pip install requests") from exc


FIXED_QUESTION = (
    "Erzaehle in einfacher Sprache eine kurze zusammenhaengende Geschichte "
    "mit klaren Saetzen und einem freundlichen Ende."
)

NO_THINKING_INSTRUCTION = (
    "Wichtig: Keine Denkprozess-Ausgabe, keine <think>-Tags, "
    "keine Analyse. Gib nur die finale Geschichte aus."
)

READ_ME_SECTION_START = "<!-- LLAMA_SERVER_BENCHMARK_RESULTS_START -->"
READ_ME_SECTION_END = "<!-- LLAMA_SERVER_BENCHMARK_RESULTS_END -->"

LOAD_RE = re.compile(
    r"(?:load time|model loaded(?: in)?|loaded in)\D*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)",
    re.IGNORECASE,
)
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*([0-9]+)\s*(?:tokens?|runs?)"
    r"(?:.*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second)?",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"eval time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*/\s*([0-9]+)\s*(?:tokens?|runs?)"
    r"(?:.*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second)?",
    re.IGNORECASE,
)


@dataclass
class RunRecord:
    model_name: str
    model_path: str
    context_target: int
    repetition: int
    port: int
    ctx_size: int
    prompt_tokens_actual: int
    output_tokens_requested: int
    output_tokens_actual: int
    load_time_s: float
    load_time_log_ms: Optional[float]
    ttft_s: float
    prompt_eval_ms: Optional[float]
    prompt_eval_tps: Optional[float]
    eval_ms: Optional[float]
    eval_tps: Optional[float]
    output_file: str
    run_log_file: str
    human_rating: Optional[float]
    human_rating_note: str
    notes: str


@dataclass
class SummaryRecord:
    model_name: str
    model_path: str
    context_target: int
    ctx_size_reserved: int
    runs: int
    prompt_tokens_mean: float
    prompt_tokens_std: float
    output_tokens_mean: float
    output_tokens_std: float
    load_time_mean_s: float
    load_time_std_s: float
    ttft_mean_ms: float
    ttft_std_ms: float
    prompt_eval_mean_ms: float
    prompt_eval_std_ms: float
    prompt_eval_tps_mean: float
    prompt_eval_tps_std: float
    eval_mean_ms: float
    eval_std_ms: float
    eval_tps_mean: float
    eval_tps_std: float
    human_rating_mean: float
    human_rating_std: float
    human_rating_samples: int
    human_rating_note: str
    notes: str


@dataclass
class RatingEntry:
    rating: float
    note: str
    updated_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GGUF models through llama-server with repeated runs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more GGUF model paths.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="llama-server host.")
    parser.add_argument("--base-port", type=int, default=11343, help="Base port for configs.")
    parser.add_argument(
        "--llama-server-bin",
        default="llama-server",
        help="Path or command name of llama-server binary.",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[64, 256, 1024, 4096],
        help="Target input token lengths.",
    )
    parser.add_argument("--output-tokens", type=int, default=256, help="Requested output tokens.")
    parser.add_argument("--repetitions", type=int, default=10, help="Runs per configuration.")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--threads-batch", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--api-mode",
        choices=["completion", "chat"],
        default="completion",
        help="Use /completion or /v1/chat/completions.",
    )
    parser.add_argument("--readme-path", default="README.md", help="README path to update.")
    parser.add_argument("--output-dir", default="llama_server_benchmarks", help="Output directory.")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument(
        "--ctx-headroom",
        type=int,
        default=128,
        help="Extra ctx-size tokens beyond input+output to avoid clipping.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort benchmark if one run fails.",
    )
    parser.add_argument(
        "--prompt-for-rating",
        action="store_true",
        help="Prompt for a human quality rating per model/context configuration.",
    )
    parser.add_argument(
        "--rating-min",
        type=float,
        default=1.0,
        help="Minimum allowed human rating value.",
    )
    parser.add_argument(
        "--rating-max",
        type=float,
        default=5.0,
        help="Maximum allowed human rating value.",
    )
    parser.add_argument(
        "--ratings-json",
        default="",
        help="Optional JSON file for loading/saving human ratings.",
    )
    parser.add_argument(
        "--rating-preview-chars",
        type=int,
        default=1800,
        help="How many characters from run_01 output to print during rating prompt.",
    )
    return parser.parse_args()


def resolve_binary(binary: str) -> str:
    if os.path.sep in binary:
        path = Path(binary)
        if not path.exists():
            raise FileNotFoundError(f"llama-server binary not found: {binary}")
        return str(path)

    resolved = shutil_which(binary)
    if resolved is None:
        raise FileNotFoundError(f"llama-server binary not found in PATH: {binary}")
    return resolved


def shutil_which(command: str) -> Optional[str]:
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(folder) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def slugify_model_name(model_path: str) -> str:
    name = Path(model_path).stem
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "model"


def rating_key(model_name: str, context_target: int) -> str:
    return f"{model_name}|ctx={context_target}"


def load_ratings(path: Path) -> Dict[str, RatingEntry]:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    ratings: Dict[str, RatingEntry] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        rating = value.get("rating")
        if not isinstance(rating, (int, float)):
            continue
        note = str(value.get("note", ""))
        updated_at = str(value.get("updated_at", ""))
        ratings[str(key)] = RatingEntry(rating=float(rating), note=note, updated_at=updated_at)

    return ratings


def save_ratings(path: Path, ratings: Dict[str, RatingEntry]) -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for key, entry in ratings.items():
        payload[key] = {
            "rating": entry.rating,
            "note": entry.note,
            "updated_at": entry.updated_at,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prompt_for_rating(
    model_name: str,
    context_target: int,
    outputs_dir: Path,
    rating_min: float,
    rating_max: float,
    preview_chars: int,
    existing: Optional[RatingEntry],
) -> Optional[RatingEntry]:
    if not sys.stdin.isatty():
        print("Skipping rating prompt because stdin is not interactive.")
        return existing

    print("\n" + "-" * 88)
    print(f"Manual rating requested for model={model_name}, input_target={context_target}")
    print(f"Output files: {outputs_dir}")

    sample_file = outputs_dir / "run_01.txt"
    if sample_file.exists():
        sample_text = sample_file.read_text(encoding="utf-8")
        preview = sample_text[:preview_chars]
        print("\nPreview of run_01 output:")
        print("~" * 40)
        print(preview.strip())
        if len(sample_text) > preview_chars:
            print("\n[truncated preview]")
        print("~" * 40)

    if existing is not None:
        print(f"Existing rating: {existing.rating:.2f} ({existing.note})")

    while True:
        raw = input(
            f"Enter rating [{rating_min}-{rating_max}] "
            "(blank keeps existing, 'skip' leaves unchanged): "
        ).strip()

        if raw == "":
            return existing
        if raw.lower() in {"skip", "s"}:
            return existing

        try:
            value = float(raw)
        except ValueError:
            print("Invalid rating. Please enter a numeric value.")
            continue

        if value < rating_min or value > rating_max:
            print("Rating out of range.")
            continue

        note = input("Optional rating note: ").strip()
        return RatingEntry(
            rating=value,
            note=note,
            updated_at=datetime.now().isoformat(timespec="seconds"),
        )


class LogCollector:
    def __init__(self, process: subprocess.Popen, log_path: Path):
        self.process = process
        self.log_path = log_path
        self._lines: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._handle = log_path.open("w", encoding="utf-8")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._handle.flush()
        self._handle.close()

    def _reader(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return

        for raw in iter(stream.readline, ""):
            if self._stop.is_set():
                break
            line = raw.rstrip("\n")
            stamped = f"[{datetime.now().isoformat(timespec='seconds')}] {line}"
            with self._lock:
                self._lines.append(stamped)
            self._handle.write(stamped + "\n")
            self._handle.flush()

    def line_count(self) -> int:
        with self._lock:
            return len(self._lines)

    def lines_since(self, index: int) -> List[str]:
        with self._lock:
            return list(self._lines[index:])

    def all_lines(self) -> List[str]:
        with self._lock:
            return list(self._lines)


def wait_for_server_ready(base_url: str, timeout_s: float) -> float:
    start = time.perf_counter()
    endpoints = ["/health", "/v1/models", "/"]

    while (time.perf_counter() - start) < timeout_s:
        for endpoint in endpoints:
            try:
                response = requests.get(base_url + endpoint, timeout=1.5)
                if response.status_code < 500:
                    return time.perf_counter() - start
            except requests.RequestException:
                continue
        time.sleep(0.2)

    raise TimeoutError(f"llama-server not ready after {timeout_s:.1f}s at {base_url}")


def parse_load_time_from_logs(log_lines: Sequence[str]) -> Optional[float]:
    candidates_ms: List[float] = []
    for line in log_lines:
        m = LOAD_RE.search(line)
        if not m:
            continue
        value = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "s":
            value *= 1000.0
        candidates_ms.append(value)
    return candidates_ms[-1] if candidates_ms else None


def parse_eval_metrics(lines: Sequence[str]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
    prompt_ms = None
    prompt_tps = None
    eval_ms = None
    eval_tps = None
    notes: List[str] = []

    for line in lines:
        m = PROMPT_EVAL_RE.search(line)
        if m:
            prompt_ms = float(m.group(1))
            prompt_tokens = int(m.group(2))
            if m.group(3):
                prompt_tps = float(m.group(3))
            elif prompt_ms > 0:
                prompt_tps = prompt_tokens / (prompt_ms / 1000.0)
                notes.append("prompt_tps_computed")

        m2 = EVAL_RE.search(line)
        if m2:
            eval_ms = float(m2.group(1))
            eval_tokens = int(m2.group(2))
            if m2.group(3):
                eval_tps = float(m2.group(3))
            elif eval_ms > 0:
                eval_tps = eval_tokens / (eval_ms / 1000.0)
                notes.append("eval_tps_computed")

    if prompt_ms is None:
        notes.append("missing_prompt_eval")
    if eval_ms is None:
        notes.append("missing_eval")

    return prompt_ms, prompt_tps, eval_ms, eval_tps, ",".join(sorted(set(notes)))


def try_count_tokens_via_server(base_url: str, text: str) -> Optional[int]:
    payloads = [
        {"content": text},
        {"prompt": text},
        {"content": text, "add_special": False},
        {"prompt": text, "add_special": False},
    ]

    for payload in payloads:
        try:
            response = requests.post(base_url + "/tokenize", json=payload, timeout=5)
            if response.status_code != 200:
                continue
            data = response.json()
        except (requests.RequestException, ValueError):
            continue

        if isinstance(data, dict):
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
            n_tokens = data.get("n_tokens")
            if isinstance(n_tokens, int):
                return n_tokens

    return None


def estimate_tokens_fallback(text: str) -> int:
    # Conservative fallback if /tokenize is unavailable.
    words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, int(round(len(words) * 1.2)))


def token_counter_factory(base_url: str) -> Tuple[Callable[[str], int], str]:
    def count_tokens(text: str) -> int:
        from_server = try_count_tokens_via_server(base_url, text)
        if from_server is not None:
            return from_server
        return estimate_tokens_fallback(text)

    probe = try_count_tokens_via_server(base_url, "Hallo Welt.")
    note = "tokenizer=server" if probe is not None else "tokenizer=fallback_estimate"
    return count_tokens, note


def sentence_pool() -> List[str]:
    return [
        "Heute klingelt der Wecker frueh und ich mache mich langsam fertig.",
        "Beim Fruehstueck gibt es Brot mit Kaese und ein Glas Milch.",
        "Auf dem Weg zur Schule winke ich meinem Nachbarn freundlich zu.",
        "Im Klassenzimmer sitzen wir zusammen und hoeren der Lehrerin zu.",
        "In der Pause spielen wir Fangen und lachen laut auf dem Hof.",
        "Meine Freundin zeigt mir ein neues Bild mit bunten Stiften.",
        "Nach dem Unterricht fahre ich vorsichtig mit dem Fahrrad nach Hause.",
        "Zu Hause wartet unser Hund und wedelt mit dem Schwanz.",
        "Wir gehen kurz in den Park und werfen einen kleinen Ball.",
        "Am Nachmittag male ich ein Haus mit Garten und Sonne.",
        "Wenn es regnet, bauen wir drinnen eine Decke als kleine Hoehle.",
        "Manchmal backen wir einfache Muffins und teilen sie mit allen.",
        "Am Wochenende besuchen wir oft Oma und Opa und trinken Kakao.",
        "Opa erzaehlt ruhige Geschichten aus seiner Kindheit.",
        "Oma zeigt mir, wie man Blumen in den Garten setzt.",
        "Abends lesen wir noch eine kurze Geschichte vor dem Schlafen.",
        "In der Schule ueben wir nette Worte und helfen einander.",
        "Beim Sport springen wir ueber Linien und zaehlen gemeinsam.",
        "Im Kunstunterricht schneiden wir Papier und kleben Sterne.",
        "Unser Kater schlaeft gern auf dem warmen Sofa.",
        "Wenn die Sonne scheint, trinken wir draussen Wasser und ruhen kurz.",
        "An kalten Tagen ziehe ich eine Muetze und Handschuhe an.",
        "Nach den Hausaufgaben spiele ich mit Bausteinen auf dem Teppich.",
        "Wir bauen einen Turm und feiern, wenn er stehen bleibt.",
        "Im Supermarkt suchen wir Obst fuer den naechsten Tag.",
        "Ich waehle einen roten Apfel und eine Banane aus.",
        "Ab und zu fahren wir mit dem Bus in die Stadtbibliothek.",
        "Dort lese ich Bilderbuecher ueber Tiere und Abenteuer.",
        "Mein kleiner Bruder lernt neue Woerter und freut sich sehr.",
        "Wenn jemand traurig ist, hoeren wir zu und troesten freundlich.",
        "Bei einem Gewitter bleiben wir drinnen und spielen Brettspiele.",
        "Nach dem Regen suchen wir kleine Pfuetzen auf dem Weg.",
        "Wir sprechen ueber gute Ideen fuer einen schoeneren Schulhof.",
        "Im Musikunterricht singen wir ein ruhiges Lied zusammen.",
        "Zum Abendessen gibt es Suppe und frisches Brot.",
        "Vor dem Schlafen packe ich den Rucksack fuer morgen.",
        "Ich lege Heft, Stifte und Trinkflasche ordentlich hinein.",
        "Morgens begruesse ich alle mit einem freundlichen Hallo.",
        "Beim Zeichnen male ich gern Tiere mit lustigen Gesichtern.",
        "Im Garten beobachten wir Schmetterlinge auf den Blumen.",
        "Wir achten darauf, langsam zu sprechen und deutlich zu erklaeren.",
        "Auf dem Spielplatz rutschen wir nacheinander und warten fair.",
        "Zu Hause giessen wir die Pflanzen mit einer kleinen Kanne.",
        "Wenn Freunde zu Besuch kommen, teilen wir unsere Spielsachen.",
        "Beim Lernen machen wir kurze Pausen und atmen tief durch.",
        "Nach der Pause klappt das Rechnen oft viel besser.",
        "Im Winter bauen wir manchmal einen kleinen Schneemann.",
        "Im Sommer fahren wir mit Helmen sicher Fahrrad.",
        "Am Ende des Tages sind wir muede und zufrieden.",
    ]


def build_german_prompt(
    target_tokens: int,
    output_tokens: int,
    ctx_size: int,
    count_tokens: Callable[[str], int],
) -> Tuple[str, int, str]:
    pool = sentence_pool()
    question_block = (
        "\n\nFrage:\n"
        + FIXED_QUESTION
        + "\n"
        + NO_THINKING_INSTRUCTION
        + "\n\nAntwort:\n"
    )

    # Keep safety room for generation and BOS/EOS behavior.
    max_prompt_tokens = max(16, ctx_size - output_tokens - 16)
    effective_target = min(target_tokens, max_prompt_tokens)

    selected: List[str] = []
    i = 0
    prompt = ""
    prompt_tokens = 0

    # Grow context with full sentences.
    while i < 20000:
        candidate_sentences = selected + [pool[i % len(pool)]]
        context = " ".join(candidate_sentences)
        candidate_prompt = context + question_block
        candidate_tokens = count_tokens(candidate_prompt)

        if candidate_tokens <= effective_target:
            selected = candidate_sentences
            prompt = candidate_prompt
            prompt_tokens = candidate_tokens
            i += 1
            continue
        break

    # Fine tune with words from one extra sentence for closer fit.
    if prompt_tokens < effective_target:
        extra_words = pool[(i + 7) % len(pool)].split()
        fine_context = " ".join(selected)
        for word in extra_words:
            candidate = (fine_context + " " + word).strip() + question_block
            candidate_tokens = count_tokens(candidate)
            if candidate_tokens <= effective_target:
                fine_context = (fine_context + " " + word).strip()
                prompt = candidate
                prompt_tokens = candidate_tokens
            else:
                break

    if not prompt:
        prompt = question_block
        prompt_tokens = count_tokens(prompt)

    delta = prompt_tokens - target_tokens
    note = f"prompt_target_delta={delta}"
    return prompt, prompt_tokens, note


def extract_stream_text_piece(api_mode: str, payload: Dict) -> str:
    if api_mode == "chat":
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict):
                return str(delta.get("content") or "")
        return ""

    # /completion mode
    if "content" in payload and isinstance(payload["content"], str):
        return payload["content"]

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        text = choices[0].get("text")
        if isinstance(text, str):
            return text

    return ""


def strip_thinking_traces(text: str) -> Tuple[str, bool]:
    had_think = bool(re.search(r"<think>", text, flags=re.IGNORECASE))
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip(), had_think


def send_streaming_request(
    base_url: str,
    api_mode: str,
    prompt: str,
    output_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    timeout_s: float,
) -> Tuple[str, float, str]:
    if api_mode == "chat":
        url = base_url + "/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stream": True,
        }
    else:
        url = base_url + "/completion"
        payload = {
            "prompt": prompt,
            "n_predict": output_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stream": True,
            "cache_prompt": False,
        }

    start = time.perf_counter()
    first_token_at: Optional[float] = None
    parts: List[str] = []

    with requests.post(url, json=payload, stream=True, timeout=(10, timeout_s)) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            piece = extract_stream_text_piece(api_mode, data)
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(piece)

    end = time.perf_counter()
    if first_token_at is None:
        first_token_at = end

    output_text = "".join(parts)
    ttft_s = first_token_at - start
    return output_text, ttft_s, ""


def start_llama_server(
    llama_server_bin: str,
    model_path: str,
    host: str,
    port: int,
    ctx_size: int,
    output_tokens: int,
    threads: int,
    threads_batch: int,
    n_gpu_layers: int,
    config_log_path: Path,
    startup_timeout_s: float,
) -> Tuple[subprocess.Popen, LogCollector, float, Optional[float]]:
    cmd = [
        llama_server_bin,
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--n-predict",
        str(output_tokens),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads_batch),
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--perf",
        "--metrics",
        "--no-cache-prompt",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collector = LogCollector(process, config_log_path)
    collector.start()

    base_url = f"http://{host}:{port}"
    try:
        load_wall_s = wait_for_server_ready(base_url, startup_timeout_s)
    except Exception:
        terminate_process(process)
        collector.stop()
        raise

    load_log_ms = parse_load_time_from_logs(collector.all_lines())
    return process, collector, load_wall_s, load_log_ms


def terminate_process(process: subprocess.Popen, timeout_s: float = 10.0) -> None:
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=3)
        except Exception:
            pass


def safe_float_mean_std(values: Sequence[Optional[float]]) -> Tuple[float, float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def group_summary(records: Sequence[RunRecord]) -> List[SummaryRecord]:
    grouped: Dict[Tuple[str, int], List[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.model_name, rec.context_target), []).append(rec)

    summary_rows: List[SummaryRecord] = []
    for (model_name, context_target), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        model_path = rows[0].model_path
        notes = sorted({r.notes for r in rows if r.notes})

        prompt_mean, prompt_std = safe_float_mean_std([r.prompt_tokens_actual for r in rows])
        out_mean, out_std = safe_float_mean_std([r.output_tokens_actual for r in rows])
        load_mean, load_std = safe_float_mean_std([r.load_time_s for r in rows])
        ttft_mean_s, ttft_std_s = safe_float_mean_std([r.ttft_s for r in rows])
        pp_ms_mean, pp_ms_std = safe_float_mean_std([r.prompt_eval_ms for r in rows])
        pp_tps_mean, pp_tps_std = safe_float_mean_std([r.prompt_eval_tps for r in rows])
        tg_ms_mean, tg_ms_std = safe_float_mean_std([r.eval_ms for r in rows])
        tg_tps_mean, tg_tps_std = safe_float_mean_std([r.eval_tps for r in rows])

        rating_values = [r.human_rating for r in rows if r.human_rating is not None]
        rating_mean, rating_std = safe_float_mean_std(rating_values)
        rating_samples = len(rating_values)
        rating_notes = sorted({r.human_rating_note for r in rows if r.human_rating_note})

        summary_rows.append(
            SummaryRecord(
                model_name=model_name,
                model_path=model_path,
                context_target=context_target,
                ctx_size_reserved=rows[0].ctx_size,
                runs=len(rows),
                prompt_tokens_mean=prompt_mean,
                prompt_tokens_std=prompt_std,
                output_tokens_mean=out_mean,
                output_tokens_std=out_std,
                load_time_mean_s=load_mean,
                load_time_std_s=load_std,
                ttft_mean_ms=ttft_mean_s * 1000.0,
                ttft_std_ms=ttft_std_s * 1000.0,
                prompt_eval_mean_ms=pp_ms_mean,
                prompt_eval_std_ms=pp_ms_std,
                prompt_eval_tps_mean=pp_tps_mean,
                prompt_eval_tps_std=pp_tps_std,
                eval_mean_ms=tg_ms_mean,
                eval_std_ms=tg_ms_std,
                eval_tps_mean=tg_tps_mean,
                eval_tps_std=tg_tps_std,
                human_rating_mean=rating_mean,
                human_rating_std=rating_std,
                human_rating_samples=rating_samples,
                human_rating_note="; ".join(rating_notes),
                notes="; ".join(notes),
            )
        )

    return summary_rows


def fmt_mean_std(mean: float, std: float, precision: int = 2) -> str:
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def print_console_table(summary_rows: Sequence[SummaryRecord]) -> None:
    headers = [
        "Model",
        "Input tgt",
        "Input actual (mu+/-sigma)",
        "Output actual (mu+/-sigma)",
        "Load s (mu+/-sigma)",
        "TTFT ms (mu+/-sigma)",
        "PP tok/s (mu+/-sigma)",
        "TG tok/s (mu+/-sigma)",
        "Human rating (mu+/-sigma)",
        "Notes",
    ]

    table_rows: List[List[str]] = []
    for s in summary_rows:
        table_rows.append(
            [
                s.model_name,
                str(s.context_target),
                fmt_mean_std(s.prompt_tokens_mean, s.prompt_tokens_std, 1),
                fmt_mean_std(s.output_tokens_mean, s.output_tokens_std, 1),
                fmt_mean_std(s.load_time_mean_s, s.load_time_std_s, 3),
                fmt_mean_std(s.ttft_mean_ms, s.ttft_std_ms, 2),
                fmt_mean_std(s.prompt_eval_tps_mean, s.prompt_eval_tps_std, 2),
                fmt_mean_std(s.eval_tps_mean, s.eval_tps_std, 2),
                fmt_mean_std(s.human_rating_mean, s.human_rating_std, 2),
                s.notes or "-",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(parts: Sequence[str]) -> str:
        return " | ".join(part.ljust(widths[i]) for i, part in enumerate(parts))

    print("\n" + line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(line(row))
    print()


def write_raw_runs_csv(path: Path, records: Sequence[RunRecord]) -> None:
    fieldnames = [
        "model_name",
        "model_path",
        "context_target",
        "repetition",
        "port",
        "ctx_size",
        "prompt_tokens_actual",
        "output_tokens_requested",
        "output_tokens_actual",
        "load_time_s",
        "load_time_log_ms",
        "ttft_s",
        "prompt_eval_ms",
        "prompt_eval_tps",
        "eval_ms",
        "eval_tps",
        "output_file",
        "run_log_file",
        "human_rating",
        "human_rating_note",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)


def write_summary_csv(path: Path, rows: Sequence[SummaryRecord]) -> None:
    fieldnames = list(SummaryRecord.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_json_summary(
    path: Path,
    raw: Sequence[RunRecord],
    summary: Sequence[SummaryRecord],
    args: argparse.Namespace,
    ratings: Dict[str, RatingEntry],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arguments": vars(args),
        "runs": [r.__dict__ for r in raw],
        "summary": [s.__dict__ for s in summary],
        "ratings": {
            key: {
                "rating": value.rating,
                "note": value.note,
                "updated_at": value.updated_at,
            }
            for key, value in ratings.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown_results_section(summary_rows: Sequence[SummaryRecord], args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.append("## Benchmark Results")
    lines.append("")
    lines.append("### General Benchmark Parameters")
    lines.append("- Benchmarked through llama-server.")
    lines.append(f"- Context targets: {', '.join(str(x) for x in args.context_lengths)} tokens.")
    lines.append(f"- Output token target: {args.output_tokens} tokens.")
    lines.append(f"- Repetitions per context: {args.repetitions}.")
    lines.append(f"- API mode: {args.api_mode}.")
    lines.append(
        "- Deterministic decode settings: "
        f"temperature={args.temperature}, top-k={args.top_k}, top-p={args.top_p}."
    )
    lines.append(
        "- Reserved llama.cpp context size is computed as: "
        f"ctx-size = input_target + output_tokens + ctx_headroom ({args.ctx_headroom})."
    )
    lines.append("- Prompt requests no reasoning trace or <think> tags.")
    lines.append("- Any remaining <think> traces are stripped from saved outputs before rating.")
    lines.append("- TTFT measured client-side; PP and TG parsed from llama-server timing logs.")
    lines.append("")

    grouped: Dict[str, List[SummaryRecord]] = {}
    for row in summary_rows:
        grouped.setdefault(row.model_name, []).append(row)

    for model_name in sorted(grouped.keys()):
        rows = sorted(grouped[model_name], key=lambda r: r.context_target)
        model_path = rows[0].model_path

        load_values = [r.load_time_mean_s for r in rows if not math.isnan(r.load_time_mean_s)]
        load_mean, load_std = safe_float_mean_std(load_values)

        rating_values = [
            r.human_rating_mean
            for r in rows
            if r.human_rating_samples > 0 and not math.isnan(r.human_rating_mean)
        ]
        rating_mean, rating_std = safe_float_mean_std(rating_values)

        lines.append(f"### {model_name}")
        lines.append(f"- Model path: {model_path}")
        lines.append(f"- Load time across context benchmarks (s): {fmt_mean_std(load_mean, load_std, 3)}")
        if rating_values:
            lines.append(f"- Human rating across contexts: {fmt_mean_std(rating_mean, rating_std, 2)}")
        else:
            lines.append("- Human rating across contexts: n/a")
        lines.append("")
        lines.append("| Context target | Reserved ctx-size | TTFT ms mean +/- std | PP tok/s mean +/- std | TG tok/s mean +/- std |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r.context_target} | {r.ctx_size_reserved} | "
                f"{fmt_mean_std(r.ttft_mean_ms, r.ttft_std_ms, 2)} | "
                f"{fmt_mean_std(r.prompt_eval_tps_mean, r.prompt_eval_tps_std, 2)} | "
                f"{fmt_mean_std(r.eval_tps_mean, r.eval_tps_std, 2)} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def update_readme(readme_path: Path, summary_rows: Sequence[SummaryRecord], args: argparse.Namespace) -> None:
    section_body = build_markdown_results_section(summary_rows, args)
    managed_block = (
        f"{READ_ME_SECTION_START}\n"
        f"{section_body}\n"
        f"{READ_ME_SECTION_END}\n"
    )

    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = ""

    if READ_ME_SECTION_START in content and READ_ME_SECTION_END in content:
        pattern = re.compile(
            re.escape(READ_ME_SECTION_START) + r".*?" + re.escape(READ_ME_SECTION_END),
            re.DOTALL,
        )
        updated = pattern.sub(managed_block.strip(), content)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        updated = content + "\n" + managed_block

    # Remove legacy block previously written by the standalone rating script.
    legacy_pattern = re.compile(
        r"\n*<!-- BEGIN RATING SUMMARY -->.*?<!-- END RATING SUMMARY -->\n*",
        re.DOTALL,
    )
    updated = legacy_pattern.sub("\n\n", updated)
    updated = re.sub(r"\n{3,}", "\n\n", updated).rstrip() + "\n"

    readme_path.write_text(updated, encoding="utf-8")


def benchmark_configuration(
    args: argparse.Namespace,
    llama_server_bin: str,
    model_path: str,
    model_name: str,
    target_input_tokens: int,
    port: int,
    output_root: Path,
    config_rating: Optional[RatingEntry],
) -> List[RunRecord]:
    ctx_size = target_input_tokens + args.output_tokens + args.ctx_headroom
    cfg_dir = output_root / model_name / f"ctx_{target_input_tokens}"
    logs_dir = cfg_dir / "logs"
    outputs_dir = cfg_dir / "outputs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{port}"
    server_log_path = logs_dir / "server_full.log"

    process = None
    collector = None
    records: List[RunRecord] = []

    try:
        process, collector, load_time_s, load_time_log_ms = start_llama_server(
            llama_server_bin=llama_server_bin,
            model_path=model_path,
            host=args.host,
            port=port,
            ctx_size=ctx_size,
            output_tokens=args.output_tokens,
            threads=args.threads,
            threads_batch=args.threads_batch,
            n_gpu_layers=args.n_gpu_layers,
            config_log_path=server_log_path,
            startup_timeout_s=args.startup_timeout,
        )

        count_tokens, tokenizer_note = token_counter_factory(base_url)
        prompt, prompt_tokens, prompt_note = build_german_prompt(
            target_tokens=target_input_tokens,
            output_tokens=args.output_tokens,
            ctx_size=ctx_size,
            count_tokens=count_tokens,
        )

        # Warmup run (not included in measured repetitions).
        warmup_note = ""
        try:
            _warmup_text, _warmup_ttft, _ = send_streaming_request(
                base_url=base_url,
                api_mode=args.api_mode,
                prompt=prompt,
                output_tokens=min(32, args.output_tokens),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                timeout_s=args.request_timeout,
            )
        except Exception as exc:
            warmup_note = f"warmup_error={type(exc).__name__}:{exc}"
            if args.stop_on_error:
                raise

        for rep in range(1, args.repetitions + 1):
            run_note_parts = [tokenizer_note, prompt_note]
            if warmup_note:
                run_note_parts.append(warmup_note)
            start_index = collector.line_count()
            output_text = ""
            ttft_s = float("nan")
            request_note = ""

            try:
                output_text, ttft_s, request_note = send_streaming_request(
                    base_url=base_url,
                    api_mode=args.api_mode,
                    prompt=prompt,
                    output_tokens=args.output_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    timeout_s=args.request_timeout,
                )
            except Exception as exc:
                request_note = f"request_error={type(exc).__name__}:{exc}"
                if args.stop_on_error:
                    raise

            # Let logger flush run-specific timing lines.
            time.sleep(0.2)
            run_lines = collector.lines_since(start_index)
            run_log_path = logs_dir / f"run_{rep:02d}.log"
            run_log_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")

            prompt_eval_ms, prompt_eval_tps, eval_ms, eval_tps, parse_note = parse_eval_metrics(run_lines)
            if parse_note:
                run_note_parts.append(parse_note)
            if request_note:
                run_note_parts.append(request_note)

            cleaned_output_text, had_think = strip_thinking_traces(output_text)
            if had_think:
                run_note_parts.append("thinking_trace_removed")

            output_tokens_actual = count_tokens(cleaned_output_text) if cleaned_output_text else 0
            out_file = outputs_dir / f"run_{rep:02d}.txt"
            out_file.write_text(cleaned_output_text, encoding="utf-8")

            record = RunRecord(
                model_name=model_name,
                model_path=str(model_path),
                context_target=target_input_tokens,
                repetition=rep,
                port=port,
                ctx_size=ctx_size,
                prompt_tokens_actual=prompt_tokens,
                output_tokens_requested=args.output_tokens,
                output_tokens_actual=output_tokens_actual,
                load_time_s=load_time_s,
                load_time_log_ms=load_time_log_ms,
                ttft_s=ttft_s,
                prompt_eval_ms=prompt_eval_ms,
                prompt_eval_tps=prompt_eval_tps,
                eval_ms=eval_ms,
                eval_tps=eval_tps,
                output_file=str(out_file),
                run_log_file=str(run_log_path),
                human_rating=(config_rating.rating if config_rating is not None else None),
                human_rating_note=(config_rating.note if config_rating is not None else ""),
                notes=";".join([p for p in run_note_parts if p]),
            )
            records.append(record)

            print(
                f"[{model_name} ctx={target_input_tokens} rep={rep}/{args.repetitions}] "
                f"prompt={prompt_tokens} out={output_tokens_actual} "
                f"load={load_time_s:.3f}s ttft={ttft_s * 1000.0:.1f}ms"
            )

    finally:
        if process is not None:
            terminate_process(process)
        if collector is not None:
            collector.stop()

    return records


def main() -> int:
    args = parse_args()

    if args.temperature != 0.0 or args.top_k != 1 or args.top_p != 1.0:
        print(
            "WARNING: deterministic benchmark fairness recommends "
            "temperature=0, top-k=1, top-p=1.",
            file=sys.stderr,
        )

    if args.rating_min >= args.rating_max:
        raise ValueError("--rating-min must be smaller than --rating-max")

    llama_server_bin = resolve_binary(args.llama_server_bin)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ratings_path = Path(args.ratings_json).resolve() if args.ratings_json else (output_root / "ratings.json")
    ratings = load_ratings(ratings_path)

    all_records: List[RunRecord] = []
    config_index = 0

    for model_path in args.models:
        model_name = slugify_model_name(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        for target_ctx in args.context_lengths:
            port = args.base_port + config_index
            config_index += 1
            cfg_key = rating_key(model_name, target_ctx)
            existing_rating = ratings.get(cfg_key)

            print("\n" + "=" * 88)
            print(
                f"Benchmarking model={model_name} target_ctx={target_ctx} "
                f"output_tokens={args.output_tokens} repetitions={args.repetitions} port={port}"
            )
            print("=" * 88)

            records = benchmark_configuration(
                args=args,
                llama_server_bin=llama_server_bin,
                model_path=model_path,
                model_name=model_name,
                target_input_tokens=target_ctx,
                port=port,
                output_root=output_root,
                config_rating=existing_rating,
            )

            if args.prompt_for_rating:
                cfg_outputs_dir = output_root / model_name / f"ctx_{target_ctx}" / "outputs"
                chosen_rating = prompt_for_rating(
                    model_name=model_name,
                    context_target=target_ctx,
                    outputs_dir=cfg_outputs_dir,
                    rating_min=args.rating_min,
                    rating_max=args.rating_max,
                    preview_chars=args.rating_preview_chars,
                    existing=existing_rating,
                )
                if chosen_rating is not None:
                    ratings[cfg_key] = chosen_rating
                final_rating = ratings.get(cfg_key)
                for rec in records:
                    rec.human_rating = final_rating.rating if final_rating is not None else None
                    rec.human_rating_note = final_rating.note if final_rating is not None else ""

            all_records.extend(records)

    save_ratings(ratings_path, ratings)
    summary_rows = group_summary(all_records)

    raw_csv = output_root / "runs_raw.csv"
    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"

    write_raw_runs_csv(raw_csv, all_records)
    write_summary_csv(summary_csv, summary_rows)
    write_json_summary(summary_json, all_records, summary_rows, args, ratings)

    print_console_table(summary_rows)

    readme_path = Path(args.readme_path)
    update_readme(readme_path, summary_rows, args)

    print(f"Raw CSV:     {raw_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"JSON:        {summary_json}")
    print(f"Ratings:     {ratings_path}")
    print(f"README:      {readme_path.resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
