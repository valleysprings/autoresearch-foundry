from __future__ import annotations

import argparse
import codecs
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ITEMS_DIR = DATA_DIR / "items"
MANIFEST_PATH = DATA_DIR / "questions.json"
FULL_DATASET_SIZE = 503
RAW_URL = "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json"
CHOICE_LABELS = ("A", "B", "C", "D")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a local LongBench v2 manifest prefix.")
    parser.add_argument("--items", type=int, default=FULL_DATASET_SIZE)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    temp_path.replace(path)


def _load_existing_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    payload = json.loads(MANIFEST_PATH.read_text())
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"items": payload}
    return None


def _iter_remote_rows(limit: int) -> Iterator[dict[str, Any]]:
    request = urllib.request.Request(
        RAW_URL,
        headers={"User-Agent": "autoresearcher/1.0"},
    )
    decoder = json.JSONDecoder()
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    yielded = 0
    buffer = ""
    position = 0
    started = False

    with urllib.request.urlopen(request, timeout=120) as response:
        while True:
            chunk = response.read(65536)
            if chunk:
                buffer += utf8_decoder.decode(chunk)
            else:
                buffer += utf8_decoder.decode(b"", final=True)

            while True:
                length = len(buffer)
                while position < length and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= length:
                        break
                    if buffer[position] != "[":
                        raise ValueError("LongBench v2 payload is not a JSON array.")
                    started = True
                    position += 1
                    continue
                while position < length and buffer[position].isspace():
                    position += 1
                if position < length and buffer[position] == ",":
                    position += 1
                    continue
                while position < length and buffer[position].isspace():
                    position += 1
                if position < length and buffer[position] == "]":
                    return
                try:
                    row, next_position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if not isinstance(row, dict):
                    raise ValueError("LongBench v2 item is not a JSON object.")
                yield row
                yielded += 1
                if yielded >= limit:
                    return
                position = next_position
                if position > 262144:
                    buffer = buffer[position:]
                    position = 0

            if not chunk:
                return


def _choice_texts(row: dict[str, Any]) -> list[str]:
    return [str(row[f"choice_{label}"]).strip() for label in CHOICE_LABELS]


def _correct_choice_index(answer_label: str) -> int:
    try:
        return CHOICE_LABELS.index(answer_label)
    except ValueError as exc:
        raise ValueError(f"LongBench v2 answer {answer_label!r} is not one of {CHOICE_LABELS}.") from exc


def main() -> None:
    args = _parse_args()
    requested_items = max(1, min(int(args.items or FULL_DATASET_SIZE), FULL_DATASET_SIZE))
    existing = _load_existing_manifest()
    existing_items = list(existing.get("items") or []) if isinstance(existing, dict) else []
    if len(existing_items) >= requested_items:
        print(f"Manifest already covers {len(existing_items)} items; requested {requested_items}.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    for source_index, row in enumerate(_iter_remote_rows(requested_items)):
        answer_label = str(row.get("answer") or "").strip()
        choices = _choice_texts(row)
        correct_choice_index = _correct_choice_index(answer_label)
        correct = choices[correct_choice_index]
        item_id = f"longbench-v2-{source_index + 1:04d}"
        domain = str(row.get("domain") or "").strip()
        domain_slug = domain.lower().replace(" ", "-")
        context_file = f"items/{item_id}.json"
        _write_json(
            DATA_DIR / context_file,
            {
                "context": str(row.get("context") or ""),
            },
        )
        items.append(
            {
                "item_id": item_id,
                "name": f"{str(row.get('domain') or '').strip()} / {str(row.get('sub_domain') or '').strip()} / {source_index + 1}",
                "prompt": str(row.get("question") or "").strip(),
                "choices": choices,
                "expected_answer": correct,
                "item_file": context_file,
                "metadata": {
                    "dataset": "longbench-v2",
                    "source_split": "train",
                    "source_index": source_index,
                    "source_id": str(row.get("_id") or ""),
                    "domain": domain,
                    "sub_domain": str(row.get("sub_domain") or "").strip(),
                    "difficulty": str(row.get("difficulty") or "").strip(),
                    "length": str(row.get("length") or "").strip(),
                    "correct_choice_index": correct_choice_index,
                    "answer_aliases": [correct],
                    "context_char_count": len(str(row.get("context") or "")),
                    "runtime_split_tags": [f"domain:{domain_slug}"],
                },
            }
        )

    manifest = {
        "dataset_id": "longbench_v2",
        "split": "train",
        "dataset_size": FULL_DATASET_SIZE,
        "prepared_count": len(items),
        "items": items,
    }
    _write_json(MANIFEST_PATH, manifest)
    print(f"Wrote {len(items)} LongBench v2 items to {MANIFEST_PATH}.")


if __name__ == "__main__":
    main()
