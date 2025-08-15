import argparse
import io
import os
import sys
from typing import Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "numind/NuMarkdown-8B-Thinking"


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def try_load_model(dtype) -> Tuple[Qwen2_5_VLForConditionalGeneration, str]:
    """
    Versucht das Modell mit absteigender Präferenz zu laden:
      1) flash_attention_2 (nur mit installiertem flash-attn)
      2) sdpa (empfohlen ohne flash-attn auf CUDA)
      3) eager (immer verfügbar)
    """
    device_map = "auto"
    errors = []

    # 1) flash_attention_2
    if torch.cuda.is_available():
        try:
            log("[INFO] Versuche attn_implementation='flash_attention_2' …")
            m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                device_map=device_map,
                trust_remote_code=True,
            )
            return m, "flash_attention_2"
        except Exception as e:
            errors.append(("flash_attention_2", str(e)))
            log("[WARN] flash_attention_2 nicht verfügbar: " + str(e)[:240] + " …")

    # 2) sdpa (oder eager auf CPU)
    try:
        impl = "sdpa" if torch.cuda.is_available() else "eager"
        log(f"[INFO] Versuche attn_implementation='{impl}' …")
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            attn_implementation=impl,
            device_map=device_map,
            trust_remote_code=True,
        )
        return m, impl
    except Exception as e:
        errors.append(("sdpa/eager", str(e)))
        log("[WARN] sdpa/eager Laden fehlgeschlagen, letzter Versuch mit 'eager' explizit …")

    # 3) eager (explizit)
    try:
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            attn_implementation="eager",
            device_map=device_map,
            trust_remote_code=True,
        )
        return m, "eager"
    except Exception as e:
        errors.append(("eager", str(e)))
        msgs = "\n".join([f" - {k}: {v}" for k, v in errors])
        raise RuntimeError("Konnte Modell nicht laden. Fehler:\n" + msgs)


def load_processor_and_model() -> Tuple[AutoProcessor, Qwen2_5_VLForConditionalGeneration, str]:
    dtype = choose_dtype()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        # sinnvolle Defaults laut Model Card
        min_pixels=100 * 28 * 28,
        max_pixels=5000 * 28 * 28,
    )
    model, attn_impl = try_load_model(dtype)
    log(f"[INFO] Modell geladen auf {model.device} mit attn='{attn_impl}', dtype={dtype}")
    return processor, model, attn_impl


def pdf_page_to_pil(page, zoom: float = 2.0) -> Image.Image:
    """
    Rendert eine PDF-Seite als PIL.Image (RGB).
    zoom=2.0 ≈ 288 dpi; bei kleiner Schrift ggf. 2.5–3.0 nutzen.
    """
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return img


def extract_answer_block(text: str) -> str:
    """
    NuMarkdown kapselt die Ausgabe meist in <answer>…</answer>.
    Falls nicht vorhanden, geben wir den gesamten Text zurück.
    """
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    return text.strip()


def run_model_on_image(
    processor: AutoProcessor,
    model: Qwen2_5_VLForConditionalGeneration,
    pil_img: Image.Image,
    temperature: float = 0.0,
    max_new_tokens: int = 4000,
) -> str:
    """
    Führt das VLM mit dem Bild aus.
    Fix: greedy (do_sample=False) bei temperature <= 0, sonst Sampling.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "You are converting a single PDF page to a table only. "
                "If the page contains a table, output the table ONLY as pure HTML (<table>) "
                "with correct <thead>, <tbody>, <tr>, <th>/<td>, and preserve exact structure: "
                "column order, row order, cell text, and any merged cells via colspan/rowspan. "
                "Do NOT describe anything. "
                "Do NOT add headings, comments, or extra text. "
                "Return ONLY inside <answer>...</answer>."
            )},
            {"type": "image"}
        ],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=prompt, images=[pil_img], return_tensors="pt")
    # Auf passendes Device schieben
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(model.device)

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

    # None-Werte entfernen
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    text = processor.decode(out[0], skip_special_tokens=True)
    return extract_answer_block(text)


def convert_pdf_to_md(
    pdf_path: str,
    out_path: str,
    page_sep: str = "<!-- PAGE BREAK -->",
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    zoom: float = 2.0,
    temperature: float = 0.0,
    max_new_tokens: int = 4000,
):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    processor, model, attn = load_processor_and_model()

    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    # Seitenbereich (1-basiert)
    s = 1 if start_page is None else max(1, start_page)
    e = n_pages if end_page is None else min(end_page, n_pages)
    if s > e:
        raise ValueError(f"Ungültiger Seitenbereich: start={s}, end={e}, total={n_pages}")

    all_md = []
    for i in range(s, e + 1):
        page = doc[i - 1]
        log(f"[INFO] Verarbeite Seite {i}/{n_pages} …")
        img = pdf_page_to_pil(page, zoom=zoom)
        page_md = run_model_on_image(
            processor, model, img,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        header = f"\n\n<!-- Page {i} -->\n\n"
        all_md.append(header + page_md)

    final_md = f"\n\n{page_sep}\n\n".join(all_md).strip()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    log(f"[DONE] Markdown geschrieben nach: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="PDF → Markdown mit NuMarkdown-8B-Thinking (robuster Fallback)")
    ap.add_argument("pdf", help="Pfad zur PDF-Datei")
    ap.add_argument("-o", "--out", help="Ausgabedatei (.md). Default: <pdfname>.md", default=None)
    ap.add_argument("--sep", default="<!-- PAGE BREAK -->", help="Trenner zwischen Seiten")
    ap.add_argument("--start", type=int, default=None, help="Startseite (1-basiert, inkl.)")
    ap.add_argument("--end", type=int, default=None, help="Endseite (1-basiert, inkl.)")
    ap.add_argument("--zoom", type=float, default=2.0, help="Render-Zoom (~DPI-Faktor). 2.0 ≈ 288 dpi")
    ap.add_argument("--temp", type=float, default=0.0, help="Sampling-Temperatur (0 = greedy)")
    ap.add_argument("--max-new-tokens", type=int, default=4000, help="Max. neue Tokens pro Seite")
    args = ap.parse_args()

    pdf_path = args.pdf
    out_path = args.out or os.path.splitext(pdf_path)[0] + ".md"

    convert_pdf_to_md(
        pdf_path=pdf_path,
        out_path=out_path,
        page_sep=args.sep,
        start_page=args.start,
        end_page=args.end,
        zoom=args.zoom,
        temperature=args.temp,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
