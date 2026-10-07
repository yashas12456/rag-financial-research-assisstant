# About this file

`questions.json` ships with **placeholder** questions and a
`REPLACE_WITH_YOUR_UPLOADED_FILENAME.pdf` document field. This is intentional
(see main spec section 26/29: never fabricate benchmark results).

## How to build a real evaluation set

1. Upload and index a real financial document (e.g. an annual report PDF)
   through the app or `POST /api/documents/upload` + `/index`.
2. Open the PDF yourself and write 15-30 real questions whose answers you can
   point to a specific page for. Financial-report question ideas:
   - "What was total revenue in fiscal year X?"
   - "What was the company's operating margin?"
   - "What risks does the company disclose related to supply chain?"
   - "How much did the company spend on R&D?"
   - "What was diluted earnings per share?"
3. For each question, record the exact filename you uploaded and the real
   page number where the answer appears.
4. Repeat across 3-5 documents to reach the 50-100 question target from the
   spec, mixing simple lookup questions with harder multi-fact ones.
5. Run `python scripts/evaluate.py` (or POST /api/evaluation/run) to compute
   real Recall@K / MRR / Hit Rate / latency against your own data.

Answer-quality metrics (answer_accuracy, citation_accuracy, faithfulness,
hallucination_rate) additionally require grading each generated answer --
either manually, or by writing an LLM-as-judge script that compares the
generated answer + citations against your `expected_answer` /
`expected_evidence_substring` fields. A stub for this is in
`scripts/evaluate.py::grade_answer_with_llm` -- wire in your own LLM_API_KEY
to use it.
