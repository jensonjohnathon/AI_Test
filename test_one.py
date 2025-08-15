from pathlib import Path
from io import BytesIO
import fitz  # PyMuPDF
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor

# ==== Config ====
PDF_PATH = Path(__file__).parent / "Kimball.pdf"
QUESTION = "What are Bridge Tables"
TOPK = 5
MODEL = "vidore/colpali-v1.2"
CACHE_FILE = PDF_PATH.with_suffix(".embeddings.v2.pt")  # bump cache version when format changes
# ===============

def render_pdf_pages(pdf_path: Path, dpi: int = 150, target_min_side: int = 448):
    """Convert PDF pages to resized PIL Images + extracted text."""
    images, texts = [], []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        w, h = img.size
        scale = target_min_side / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        images.append(img)
        texts.append(page.get_text("text"))
    doc.close()
    return images, texts

@torch.no_grad()
def encode_docs_colpali(model, processor, images):
    """Encode a list of PIL images into embeddings [n_vec, dim]."""
    device = next(model.parameters()).device
    embs = []
    for img in tqdm(images, desc="Encoding pages"):
        batch = processor(
            text=["<image>"],           # IMPORTANT: one <image> per image
            images=[img],
            return_tensors="pt",
            truncation=False,           # avoid token drop that breaks alignment
            padding=True,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        emb = model(**batch)            # [1, n_vec, dim] or [n_vec, dim]
        if emb.ndim == 3:
            emb = emb.squeeze(0)        # -> [n_vec, dim]
        elif emb.ndim == 1:             # safety
            emb = emb.unsqueeze(0)
        embs.append(emb.cpu())
    return embs

@torch.no_grad()
def encode_query_colpali(model, processor, query: str):
    """Encode a text query into embedding [n_vec, dim]."""
    device = next(model.parameters()).device
    blank = Image.new("RGB", (448, 448), (255, 255, 255))
    batch = processor(
        text=[f"{query} <image>"],      # query + placeholder for the blank image
        images=[blank],
        return_tensors="pt",
        truncation=False,
        padding=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    q = model(**batch)                  # [1, n_vec, dim] or [n_vec, dim]
    if q.ndim == 3:
        q = q.squeeze(0)
    elif q.ndim == 1:
        q = q.unsqueeze(0)
    return q.cpu()

def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Make sure tensor is [n_vec, dim]."""
    if x.ndim == 3:
        x = x.squeeze(0)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1])
    return x

def cosine_maxsim(qv: torch.Tensor, dv: torch.Tensor) -> float:
    """Cosine MaxSim in 0–1 range: mean over per-query-token max similarities."""
    qv = ensure_2d(qv)
    dv = ensure_2d(dv)
    qn = F.normalize(qv, p=2, dim=-1)
    dn = F.normalize(dv, p=2, dim=-1)
    sims = qn @ dn.transpose(0, 1)      # [nq, nd]
    return sims.max(dim=1).values.mean().item()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model {MODEL} on {device} ({dtype}) ...")
    model = ColPali.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL, use_fast=True)

    # Load or compute embeddings
    if CACHE_FILE.exists():
        print(f"Loading cached embeddings from {CACHE_FILE} ...")
        cache = torch.load(CACHE_FILE)
        doc_vecs = [ensure_2d(t) for t in cache["doc_vecs"]]
        page_texts = cache["page_texts"]
    else:
        print("Rendering pages ...")
        images, page_texts = render_pdf_pages(PDF_PATH)
        print(f"Encoding {len(images)} pages ...")
        doc_vecs = encode_docs_colpali(model, processor, images)
        torch.save({"doc_vecs": doc_vecs, "page_texts": page_texts}, CACHE_FILE)

    # Encode query
    print(f"Encoding query: {QUESTION!r}")
    qvec = encode_query_colpali(model, processor, QUESTION)

    # Score
    print("Scoring (cosine maxsim 0–1) ...")
    scores = [(i, cosine_maxsim(qvec, dv)) for i, dv in enumerate(doc_vecs)]
    scores.sort(key=lambda x: x[1], reverse=True)

    # Output
    print("\nTop Seiten:")
    for idx, s in scores[:TOPK]:
        snippet = (page_texts[idx] or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"  - Seite {idx+1:>3} | Score {s:.3f} | {snippet}")

if __name__ == "__main__":
    main()
from pathlib import Path
from io import BytesIO
import fitz  # PyMuPDF
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor

# ==== Config ====
PDF_PATH = Path(__file__).parent / "Kimball.pdf"
QUESTION = "What are Bridge Tables"
TOPK = 5
MODEL = "vidore/colpali-v1.2"
CACHE_FILE = PDF_PATH.with_suffix(".embeddings.pt")
# ===============


def render_pdf_pages(pdf_path: Path, dpi: int = 150, target_min_side: int = 448):
    """Convert PDF pages to resized PIL Images + extracted text."""
    images, texts = [], []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        w, h = img.size
        scale = target_min_side / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        images.append(img)
        texts.append(page.get_text("text"))
    doc.close()
    return images, texts


@torch.no_grad()
def encode_docs_colpali(model, processor, images):
    """Encode a list of PIL images into embeddings."""
    device = next(model.parameters()).device
    embs = []
    for img in tqdm(images, desc="Encoding pages"):
        batch = processor(text=[""], images=[img], return_tensors="pt", padding=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        emb = model(**batch)
        embs.append(emb.cpu())
    return embs


@torch.no_grad()
def encode_query_colpali(model, processor, query: str):
    """Encode a text query into an embedding."""
    device = next(model.parameters()).device
    blank = Image.new("RGB", (448, 448), (255, 255, 255))
    batch = processor(text=[query], images=[blank], return_tensors="pt", padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    return model(**batch).cpu()


def cosine_maxsim(qv: torch.Tensor, dv: torch.Tensor) -> float:
    """Cosine similarity with max over document tokens."""
    qn = F.normalize(qv, p=2, dim=-1)
    dn = F.normalize(dv, p=2, dim=-1)
    sims = qn @ dn.T  # shape: [query_tokens, doc_tokens]
    return sims.max(dim=1).values.mean().item()  # 0–1 range


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model {MODEL} on {device} ({dtype}) ...")
    model = ColPali.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL, use_fast=True)

    # Load or compute embeddings
    if CACHE_FILE.exists():
        print(f"Loading cached embeddings from {CACHE_FILE} ...")
        cache = torch.load(CACHE_FILE)
        doc_vecs, page_texts = cache["doc_vecs"], cache["page_texts"]
    else:
        print("Rendering pages ...")
        images, page_texts = render_pdf_pages(PDF_PATH)
        print(f"Encoding {len(images)} pages ...")
        doc_vecs = encode_docs_colpali(model, processor, images)
        torch.save({"doc_vecs": doc_vecs, "page_texts": page_texts}, CACHE_FILE)

    # Encode query
    print(f"Encoding query: {QUESTION!r}")
    qvec = encode_query_colpali(model, processor, QUESTION)

    # Score
    print("Scoring (cosine maxsim) ...")
    scores = [(i, cosine_maxsim(qvec, dv)) for i, dv in enumerate(doc_vecs)]
    scores.sort(key=lambda x: x[1], reverse=True)

    # Output
    print("\nTop Seiten:")
    for idx, s in scores[:TOPK]:
        snippet = (page_texts[idx] or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"  - Seite {idx+1:>3} | Score {s:.3f} | {snippet}")


if __name__ == "__main__":
    main()
