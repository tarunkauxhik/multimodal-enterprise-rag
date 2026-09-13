# Evaluation dataset

`gold.jsonl` is a 30-question gold set selected from the 41 verified Step 8A candidates. It was built from three PDFs in `evals/docs/` (PDFs are gitignored):

| File | Contents |
|---|---|
| `niti_2025_26_english.pdf` | NITI Aayog Annual Report 2025-26, English (196 pages) |
| `niti_2025_26_hindi.pdf` | NITI Aayog Annual Report 2025-26, Hindi edition (196 pages) |
| `mhi_2025_26_bilingual.pdf` | Ministry of Heavy Industries Annual Report 2025-26: Hindi half PDF 1-136, English half 137-266 (266 pages) |

Every answer, page and evidence snippet comes from the PDFs. `page` is the **PDF page number** (1-based), not the printed page. A list means the answer needs both pages.

## Record fields

`id`, `group`, `question`, `document`, `page`, `answer`, `evidence`, `answerable`, `query_language` (`en`, `hi`, `hinglish`), `category`, `modality` (`text`, `table`, `figure`, `scan`).

Hindi evidence is quoted exactly as PyMuPDF extracts it, including extraction damage (dropped conjunct parts, doubled vowel signs). Table evidence joins cells that sit on separate lines in the extracted text.

## Groups

| Group | Count | IDs |
|---|---|---|
| `niti_english` | 8 | E01 E02 E04 E06 T01 T03 T04 V01 |
| `niti_hindi` | 5 | H01 H02 H03 H04 H05 |
| `cross_language` | 5 | X01 X02 (English question, Hindi document); X03 X04 X05 (Hindi question, English MHI pages) |
| `hinglish` | 4 | G01 G03 G04 G05 |
| `mhi_table_multimodal` | 6 | M01 M02 M03 M04 M05 S01 |
| `unanswerable_injection` | 2 | U02 (must abstain), P01 (contains an injected instruction; must still answer with a citation) |

The requested split (8/5/4/4/5/2) adds up to 28. The two extra slots went to X04 (cross-language) and M03 (a table with merged cells).

## Categories

- `text`: fact stated in running text.
- `multi_chunk`, `table_multi_chunk`: the answer combines facts from two pages.
- `table`, `table_reasoning`: value lookup or comparison in a table. M03's extracted table has merged cells.
- `figure`: needs visual understanding. V01 is a map image; H05's page has a broken-font text layer.
- `scanned`, `hinglish_scanned`: image-only content (S01 is a Gazette scan, G05 an organogram).
- `cross_language_en_to_hi`, `cross_language_hi_to_en`: question and evidence in different languages.
- `hinglish`, `hinglish_table`: romanised Hindi questions.
- `unanswerable`: the documents do not contain the answer; the correct behaviour is to abstain.
- `prompt_injection`: the question contains an instruction the system must ignore.

## Known limitations the set avoids depending on

- The MHI Hindi half (PDF 1-136) uses legacy non-Unicode fonts (Arjun, BHARTIYA-HINDI_081) and extracts as Latin gibberish. The Hindi→English questions therefore point at the English half.
- The visual questions (V01, H05, G05, S01) can only be answered if ingestion runs selective MiniMax-M3 page understanding.
