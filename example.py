# try_one.py
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset


def main() -> None:
    """Example script to run inference with ColPali (0.3.x API)."""

    # -------- Load model & processor (no adapters needed) --------
    MODEL = "vidore/colpali-v1.2"

    # pick device & dtype
    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = ColPali.from_pretrained(MODEL, torch_dtype=dtype, device_map=device_map).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL)

    # -------- Data --------
    # Option 1: small demo dataset hosted on HF
    images = load_from_dataset("vidore/docvqa_test_subsampled")

    # Option 2: or replace with your own list of PIL Images
    # images = [Image.open("path/to/your/image1.png"), Image.open("path/to/your/image2.png")]

    queries = [
        "From which university does James V. Fiorca come?",
        "Who is the Japanese prime minister?"
    ]

    # -------- Encode docs --------
    doc_loader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: process_images(processor, batch),
    )

    ds = []
    for batch_doc in tqdm(doc_loader, desc="Encoding documents"):
        with torch.no_grad():
            # If you didn't use device_map="auto", move tensors manually:
            if device_map is None:
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    # -------- Encode queries --------
    # build a blank placeholder image for query side (as in the ColPali examples)
    blank_img = Image.new("RGB", (448, 448), (255, 255, 255))

    query_loader = DataLoader(
        queries,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: process_queries(processor, batch, blank_img),
    )

    qs = []
    for batch_query in tqdm(query_loader, desc="Encoding queries"):
        with torch.no_grad():
            if device_map is None:
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    # -------- Evaluate retrieval --------
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate(qs, ds)  # shape: [num_queries, num_docs]
    print("Argmax doc per query:", scores.argmax(axis=1))


if __name__ == "__main__":
    typer.run(main)
