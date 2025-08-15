# PDF Table Extraction & Embedding Toolkit

Dieses Projekt enthält verschiedene Python-Skripte zur Extraktion von Tabellen aus PDFs, deren Umwandlung in Markdown sowie zur Erstellung und Auswertung von Embeddings für Dokumente und Anfragen.

## Übersicht der Skripte

### [`pdf_to_md.py`](pdf_to_md.py)

- **Funktion:** Extrahiert Tabellen aus einer PDF-Datei und wandelt jede Seite in Markdown um. Nutzt ein Vision-Language-Modell (NuMarkdown-8B-Thinking).
- **Output:**
- **Beispiel:**
  ```
  python pdf_to_md.py input.pdf -o out.md
  ```

### [`f_pdf_to_md.py`](f_pdf_to_md.py)

- **Funktion:** Alternative, flexible Version von `pdf_to_md.py` mit zusätzlichen Optionen (z.B. 4-Bit-Quantisierung, Streaming, VRAM-Limits).
- **Output:** Erstellt ebenfalls eine Markdown-Datei z.B. `out1.md` und `out.md`
- **Beispiel:**
  ```
  python f_pdf_to_md.py input.pdf -o out1.md --max-pixels 1800*28*28 --quant-4bit
  ```

### [`example.py`](example.py)

- **Funktion:** Beispielskript zur Inferenz mit ColPali (API 0.3.x). Zeigt, wie Dokumente und Anfragen als Embeddings kodiert und verglichen werden.
- **Output:** Erstellt Embedding-Dateien z.B. `Kimball.embeddings.pt`, `Kimball.embeddings.v2.pt` und auch die json in `outputs/`.

### [`test_one.py`](test_one.py)

- **Funktion:** Führt eine Suche über die kodierten Dokumentseiten durch. Kodiert eine Anfrage, berechnet Ähnlichkeiten und gibt die Top-Treffer mit Seitennummer und Score aus.
- **Output:**
  - Textausgabe mit Scores und Seitennummern (z.B. `output.txt`)
  - Nutzt Embedding-Dateien als Input
  - Bis jetzt scheint aber noch ein Fehler aufzutreten da alle Seiten gleich relevant sind.
