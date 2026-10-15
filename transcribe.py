#!/usr/bin/env python3
"""Transcribe local media with oMLX; retain chunks and results for resuming."""
import argparse
import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave

from dotenv import dotenv_values


def load_config():
    env_path = pathlib.Path(__file__).resolve().parent / ".env"
    values = dotenv_values(env_path)
    endpoint = (values.get("OMLX_ENDPOINT") or "").strip().rstrip("/")
    model = (values.get("OMLX_MODEL") or "").strip()
    if not endpoint or not model or "OMLX_API_KEY" not in values:
        raise RuntimeError(
            f"Set OMLX_ENDPOINT, OMLX_API_KEY, and OMLX_MODEL in {env_path} "
            "(see .env.example; API key may be empty)."
        )
    if not endpoint.startswith(("http://", "https://")):
        raise RuntimeError("OMLX_ENDPOINT must start with http:// or https://")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    key = (values.get("OMLX_API_KEY") or "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return endpoint, headers, model


def prepare_chunks(source, out, chunk_seconds):
    """Use Silero ONNX speech probabilities to find boundaries between speech."""
    import numpy as np
    import onnxruntime as ort
    full = out / "audio.wav"
    plan_path = out / "segments.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    if not full.exists():
        partial = out / "audio.partial.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-y", "-i", str(source), "-map", "0:a:0", "-vn",
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(partial)], check=True)
        partial.replace(full)
    assets = pathlib.Path(__file__).resolve().parent / "assets"
    assets.mkdir(exist_ok=True)
    model_file = assets / "silero_vad.onnx"
    if not model_file.exists():
        url = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        import hashlib
        temp = model_file.with_suffix(".tmp")
        temp.write_bytes(payload)
        temp.replace(model_file)
        (assets / "model_source.json").write_text(json.dumps({"url": url, "sha256": hashlib.sha256(payload).hexdigest()}, indent=2))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_file), sess_options=opts, providers=["CPUExecutionProvider"])
    with wave.open(str(full), "rb") as reader:
        rate = reader.getframerate()
        total_frames = reader.getnframes()
        duration = total_frames / rate
        probs_file = out / "vad_probabilities.npy"
        if probs_file.exists():
            probs = np.load(probs_file)
        else:
            state = np.zeros((2, 1, 128), dtype=np.float32)
            context = np.zeros((1, 64), dtype=np.float32)
            probabilities = []
            started = time.monotonic()
            while True:
                raw = reader.readframes(512)
                if not raw:
                    break
                x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                if len(x) < 512:
                    x = np.pad(x, (0, 512-len(x)))
                frame = np.concatenate((context, x.reshape(1, -1)), axis=1)
                prediction, state = session.run(None, {"input": frame, "state": state, "sr": np.array(16000, dtype=np.int64)})
                context = frame[:, -64:].copy()
                probabilities.append(float(prediction.item()))
                if len(probabilities) % 50000 == 0:
                    print(f"VAD: {len(probabilities)*.032/duration:.0%}; elapsed={time.monotonic()-started:.1f}s", flush=True)
            probs = np.array(probabilities, dtype=np.float32)
            np.save(probs_file, probs)
            print(f"VAD complete: {duration:.1f}s audio in {time.monotonic()-started:.1f}s", flush=True)
        plan = []
        start = 0.0
        overlap = False
        while start < duration:
            end = min(start + chunk_seconds, duration)
            clean = True
            if end < duration:
                lo = int((start + chunk_seconds * .6) / .032)
                hi = min(int(end / .032), len(probs))
                pauses = []
                run = None
                for k in range(lo, hi):
                    if probs[k] < .35:
                        if run is None:
                            run = k
                    else:
                        if run is not None and k-run >= 5:
                            pauses.append((run+k)*.016)
                        run = None
                if run is not None and hi-run >= 5:
                    pauses.append((run+hi)*.016)
                if pauses:
                    end = pauses[-1]
                else:
                    k = min(range(lo, hi), key=lambda x: float(np.mean(probs[max(lo,x-2):min(hi,x+3)])))
                    end = k*.032
                    clean = False
            name = f"{len(plan):06d}.wav"
            reader.setpos(round(start*rate))
            data = reader.readframes(round((end-start)*rate))
            with wave.open(str(out / "chunks" / name), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(rate)
                writer.writeframes(data)
            plan.append({"file": name, "start": start, "end": end, "overlap": overlap})
            if end >= duration:
                break
            start = end if clean else end-1.0
            overlap = not clean
    plan_path.write_text(json.dumps(plan, indent=2))
    return plan


def trim_overlap(previous, current):
    # Remove only an exact matching suffix/prefix from overlapping audio.
    for n in range(min(50, len(previous), len(current)), 0, -1):
        if previous.endswith(current[:n]):
            return current[n:]
    return current


def stamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("--language", default="zh")
    p.add_argument("--chunk-seconds", type=int, default=25)
    p.add_argument("--output-dir", type=pathlib.Path)
    args = p.parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file() or args.chunk_seconds < 1:
        p.error("Input must exist and chunk duration must be positive")
    base, auth, model = load_config()
    # Avoid sending local audio through an environment-configured HTTP proxy.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(urllib.request.Request(base + "/models", headers=auth), timeout=30) as r:
        models = [m["id"] for m in json.load(r)["data"]]
    if model not in models:
        raise RuntimeError(f"Model '{model}' from .env not found. Available: {models}")
    out = args.output_dir or source.with_name(source.stem + "_asr")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"input": str(source), "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns, "model": model,
                "language": args.language, "chunk_seconds": args.chunk_seconds,
                "segmentation": "silero-v6.2.1-overlap-v1"}
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise RuntimeError("Output settings differ from previous run; use a new --output-dir")
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    chunks_dir = out / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    plan = prepare_chunks(source, out, args.chunk_seconds)
    chunks = [chunks_dir / part["file"] for part in plan]
    if not chunks:
        raise RuntimeError("No audio chunks were extracted")
    results_dir = out / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Model: {model}; chunks: {len(chunks)}; output: {out}", flush=True)
    started = time.monotonic()
    for i, chunk in enumerate(chunks):
        result_path = results_dir / (chunk.stem + ".json")
        if not result_path.exists():
            boundary = uuid.uuid4().hex
            body = bytearray()
            for name, value in {"model": model, "language": args.language}.items():
                body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
            body.extend(chunk.read_bytes())
            body.extend(f"\r\n--{boundary}--\r\n".encode())
            req = urllib.request.Request(base + "/audio/transcriptions", data=bytes(body),
                                         headers={**auth, "Content-Type": f"multipart/form-data; boundary={boundary}"})
            for attempt in range(3):
                try:
                    with client.open(req, timeout=600) as response:
                        result = json.load(response)
                    if not isinstance(result.get("text"), str):
                        raise RuntimeError(f"Unexpected response: {result}")
                    temp = result_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    temp.replace(result_path)
                    break
                except urllib.error.HTTPError as e:
                    detail = e.read().decode(errors="replace")
                    if e.code < 500 or attempt == 2:
                        raise RuntimeError(f"Chunk {i}: HTTP {e.code}: {detail}") from e
                    time.sleep(2 ** attempt)
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            print(f"[{i+1}/{len(chunks)}] {stamp(plan[i]['start'])} elapsed={time.monotonic()-started:.1f}s", flush=True)
        # Rebuild partial deliverables after every completed chunk, including resumed chunks.
        rows = []
        for j, part in enumerate(chunks):
            rp = results_dir / (part.stem + ".json")
            if not rp.exists():
                break
            text = json.loads(rp.read_text())["text"].strip()
            if plan[j]["overlap"] and rows:
                text = trim_overlap(rows[-1][1], text)
            rows.append((plan[j]["start"], text))
        (out / "transcript.txt").write_text("\n\n".join(t for _, t in rows) + "\n")
        (out / "transcript_timestamped.txt").write_text("\n\n".join(f"[{stamp(s)}]\n{t}" for s, t in rows) + "\n")
    print(f"Done: {out / 'transcript.txt'}", flush=True)


if __name__ == "__main__":
    main()
