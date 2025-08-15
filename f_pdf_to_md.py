import argparse, io, os, sys, time
from typing import Optional
import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Optional: 4-bit Quantisierung
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

MODEL_ID = "numind/NuMarkdown-8B-Thinking"

def log(msg: str): print(msg, file=sys.stderr, flush=True)

def choose_dtype(fp16: bool) -> torch.dtype:
    if torch.cuda.is_available():
        # Auf RTX 30xx ist fp16 i.d.R. schneller als bf16
        return torch.float16 if fp16 else torch.bfloat16
    return torch.float32

def load_processor(max_pixels: int):
    # min/max_pixels: steuern die interne Bildgröße → großer Hebel für VRAM/Speed
    return AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        min_pixels=100*28*28, max_pixels=max_pixels
    )

def try_load_model(dtype, quant_4bit: bool, no_kv_cache: bool, attn_pref="sdpa"):
    device_map = "auto"
    base_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)

    if quant_4bit:
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes fehlt. Installiere mit: pip install bitsandbytes")
        log("[INFO] Lade Modell in 4-Bit (bitsandbytes)…")
        qconf = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
        )
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map=("cuda" if torch.cuda.is_available() else device_map),
            attn_implementation=(attn_pref if torch.cuda.is_available() else "eager"),
            quantization_config=qconf,
            **base_kwargs
        )
        m.generation_config.use_cache = not no_kv_cache
        return m, f"4bit+{attn_pref}"

    # Ohne 4-Bit: versuche flash_attn -> sdpa -> eager
    impls = []
    if torch.cuda.is_available():
        impls += ["flash_attention_2", attn_pref]
    impls += ["eager"]

    last_err = None
    for impl in impls:
        try:
            log(f"[INFO] Versuche attn_implementation='{impl}' …")
            m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, device_map=device_map, attn_implementation=impl, **base_kwargs
            )
            m.generation_config.use_cache = not no_kv_cache
            return m, impl
        except Exception as e:
            log(f"[WARN] {impl} nicht verfügbar: {str(e)[:200]} …")
            last_err = e
    raise last_err

def pdf_page_to_pil(page, zoom: float = 2.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

def downscale_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = (max_pixels / (w * h)) ** 0.5
    new_size = (max(1, int(w * s)), max(1, int(h * s)))
    return img.resize(new_size, Image.BICUBIC)

def extract_answer_block(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return text.strip()

def generate_markdown(processor, model, pil_img, temperature: float, max_new_tokens: int, stream: bool):
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # OPTIONAL: zusätzliche Verkleinerung (bereits per max_pixels gesteuert)
    # pil_img = downscale_to_max_pixels(pil_img, max_pixels=1600*28*28)

    inputs = processor(text=prompt, images=[pil_img], return_tensors="pt")
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(model.device)

    # Greedy wenn temp <= 0, Sampling nur wenn > 0
    use_sampling = (temperature is not None) and (float(temperature) > 0.0)
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": use_sampling,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None),
        "eos_token_id": (
            model.generation_config.eos_token_id
            if model.generation_config.eos_token_id is not None
            else getattr(processor.tokenizer, "eos_token_id", None)
        ),
    }
    if use_sampling:
        gen_kwargs["temperature"] = float(temperature)
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    if stream:
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        import threading
        out_ids = []
        def _gen():
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
                out_ids.append(out)
        t = threading.Thread(target=_gen); t.start()

        last = time.time()
        for token in streamer:
            if time.time() - last > 1.0:
                log(f"[GEN] …{token}")
                last = time.time()
        t.join()
        text = processor.decode(out_ids[0][0], skip_special_tokens=True)
    else:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        text = processor.decode(out[0], skip_special_tokens=True)

    return extract_answer_block(text)

def convert_pdf_to_md(pdf_path: str, out_path: str, page_sep: str, start_page: Optional[int],
                      end_page: Optional[int], zoom: float, temperature: float, max_new_tokens: int,
                      max_pixels: int, quant_4bit: bool, fp16: bool, no_kv_cache: bool, stream: bool):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    dtype = choose_dtype(fp16)
    processor = load_processor(max_pixels)
    model, attn = try_load_model(dtype, quant_4bit=quant_4bit, no_kv_cache=no_kv_cache, attn_pref="sdpa")
    log(f"[INFO] Modell auf {model.device} geladen, attn='{attn}', dtype={dtype}, use_cache={model.generation_config.use_cache}")

    doc = fitz.open(pdf_path); n_pages = len(doc)
    s = 1 if start_page is None else max(1, start_page)
    e = n_pages if end_page is None else min(end_page, n_pages)
    if s > e: raise ValueError(f"Ungültiger Seitenbereich: start={s}, end={e}, total={n_pages}")

    parts = []
    for i in range(s, e + 1):
        page = doc[i - 1]
        log(f"[INFO] Verarbeite Seite {i}/{n_pages} …")
        img = pdf_page_to_pil(page, zoom=zoom)
        img = downscale_to_max_pixels(img, max_pixels=max_pixels)
        md = generate_markdown(processor, model, img, temperature, max_new_tokens, stream)
        parts.append(f"\n\n<!-- Page {i} -->\n\n" + md)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write((f"\n\n{page_sep}\n\n").join(parts).strip())
    log(f"[DONE] Markdown geschrieben nach: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="PDF → Markdown mit NuMarkdown-8B-Thinking (Low-VRAM/4-Bit-Optionen)")
    ap.add_argument("pdf"); ap.add_argument("-o","--out", default=None)
    ap.add_argument("--sep", default="<!-- PAGE BREAK -->")
    ap.add_argument("--start", type=int, default=None); ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--max-pixels", type=int, default=1800*28*28, help="Obergrenze für Bildpixel (W*H).")
    ap.add_argument("--quant-4bit", action="store_true", help="Lade Modell in 4-Bit (bitsandbytes).")
    ap.add_argument("--fp16", action="store_true", help="Nutze fp16 (schneller auf RTX 3060).")
    ap.add_argument("--no-kv-cache", action="store_true", help="KV-Cache aus (spart VRAM; evtl. langsamer).")
    ap.add_argument("--stream", action="store_true", help="Token-Streaming ins Log.")
    args = ap.parse_args()

    pdf_path = args.pdf
    out_path = args.out or os.path.splitext(pdf_path)[0] + ".md"

    convert_pdf_to_md(
        pdf_path, out_path, args.sep, args.start, args.end, args.zoom, args.temp,
        args.max_new_tokens, args.max_pixels, args.quant_4bit, args.fp16, args.no_kv_cache, args.stream
    )

if __name__ == "__main__":
    main()
