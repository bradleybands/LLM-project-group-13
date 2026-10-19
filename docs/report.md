# JobMatch: English access to Swedish job advertisements through multilingual RAG

**Status:** Implementation and methods draft. Live OpenAI smoke tests have run; full generation experiments and manual claim scoring must be completed before submission. This document is Markdown, not a finished university-template PDF. Replace placeholders with measured results and your own discussion.

## Abstract

JobMatch retrieves evidence from a fixed snapshot of Swedish JobTech advertisements and generates English answers with source references. It combines local multilingual embeddings and keyword retrieval with metadata filters, streaming generation and CV comparison. The implementation keeps original Swedish/English source text, uses OpenAI for generation and offers an optional Ollama backend. Experiments are designed to compare retrieval strategies, chunk sizes, grounded versus ungrounded generation, and model backends. Final conclusions require reviewed labels and measured generation results.

## Problem and scope

International students and newcomers may find Swedish job advertisements difficult to navigate. A language model answering from memory can invent employers, vacancies or requirements. We investigate whether retrieved source excerpts improve the factual basis of English explanations and skill-gap analysis. The dataset is a dated sample of Platsbanken advertisements, not a census of all jobs in Sweden. Data-science scope is based on explicit keyword matches and includes adjacent analytics/data-engineering roles.

## System

1. Download JobTech `/v2/snapshot` in JSONL and retain data-science-related ads. Preserve original descriptions and metadata. Clean HTML, normalize whitespace, remove empty/duplicate IDs and identical descriptions.
2. Use the embedding model's tokenizer to make overlapping chunks: baseline 320 tokens with 60 overlap. Add a bounded title/employer/location/occupation header.
3. Embed passages with local `intfloat/multilingual-e5-small` and persist vectors in Chroma. The query uses the same model with the appropriate `query:` prefix.
4. Combine cosine-nearest-neighbour retrieval and BM25L by reciprocal-rank fusion. Apply identical location/occupation filters before ranking and limit repeated chunks per ad.
5. Supply numbered excerpts to OpenAI Responses, streaming an English answer. Instructions prohibit unsupported claims, preserve requirements and require citations/refusal. No retrieved context produces a deterministic refusal without an API call.
6. CV mode accepts pasted text or uploaded PDFs. Native text is extracted locally with pypdf; pages with little usable text fall back to PDFium rendering and Tesseract OCR with English/Swedish language data. Users review/edit the extracted text and enable consent before it becomes model context. Structured comparisons validate source/CV quotes and do not certify proficiency, degree completion, hiring likelihood or protected traits.

The sources are untrusted input; instructions embedded in ads or CVs are not intended to override system instructions. This prompt boundary mitigates injection but is not a formal guarantee. Numeric citation validation catches nonexistent references; human scoring is still needed to verify whether a citation supports a claim.

## Experimental method

Freeze the dataset fingerprint and a reviewed 30-question set. The generated draft contains 15 specific questions, 8 filtered queries, 4 synthetic CVs and 3 unsupported requests. Exact-title questions are an easy baseline: rewrite a subset as natural paraphrases, including English questions targeting Swedish source text, to avoid overstating semantic retrieval performance. Record gold ad IDs and expected requirements directly from the original advertisements; mark reviewed labels explicitly.

Retrieval: compare BM25L, embeddings and hybrid; compare 200, 320 and 400 token chunks with proportional overlap. Measure hit@k and ad recall@k at fixed chunk k. Report specific and filter questions separately because multi-ad gold sets can exceed the retrieval budget. Embedding context prevents directly using an 800-token E5 chunk without truncation, so this implementation uses sizes below the model context limit.

Generation: run the same questions with and without RAG, keeping model/prompt conditions as comparable as possible. Record unsupported claims, supported claims, refusal correctness, first-token and total latency, token usage and cost. A no-RAG baseline necessarily lacks evidence; manually assess against gold source facts rather than whether it merely produces citations.

Optional model comparison: use the same retrieved contexts with OpenAI and an installed open model through Ollama. Note hardware and decoding differences. Execute in a controlled order and, if time allows, repeat latency measurements to reduce cache/network effects. Do not treat $0 API charge as zero computational cost.

## Results to complete

| Experiment | Required evidence | Status |
|---|---|---|
| Retrieval strategies | `evaluation/results/*/retrieval.csv` with reviewed gold labels | Preliminary draft-labelled runs completed |
| Chunking | Same questions over 200/320/400-token indexes | All three preliminary runs completed |
| RAG vs no RAG | Generation CSV plus manual unsupported-claim counts | Requires OpenAI key and scoring |
| Open vs OpenAI | Same questions/contexts, latency and faithfulness | Optional Ollama setup required |

Insert measured tables here. Do not assert reduced hallucinations or a superior backend until the experiments support those claims. Preserve failed runs and report their frequency.

The measured preliminary retrieval table is in [preliminary_results.md](preliminary_results.md). Its draft-labelled exact-title questions do not establish general retrieval performance; review and rerun before incorporating the numbers as final results.

### Live development checks

Short OpenAI checks with public ad excerpts and a synthetic CV confirmed streaming English output and API usage reporting. A hiring-guarantee question was refused. Initial free-form CV answers incorrectly inferred language requirements and treated ongoing studies or listed skills as meeting stronger requirements. Prompt revisions alone did not eliminate these errors. CV matching was therefore changed to structured quote extraction followed by literal source/CV quote validation and deterministic presentation. The display does not certify degree completion, proficiency or eligibility. Missing extraction is reported explicitly. This improves inspectability but does not guarantee translation quality, relevance, completeness or correct classification of mandatory versus preferred criteria; original quoted wording remains authoritative. Saved smoke outputs are development examples, not aggregate benchmark results.

## Societal impact and limitations

English explanations and multilingual retrieval can lower language barriers and help applicants inspect requirements with direct evidence. Citations give users a way to verify a recommendation and make missing evidence visible.

The dataset inherits selection and wording biases from advertisements and the keyword filter. CV descriptions can omit genuine ability; therefore “not evidenced in your CV” is preferred to “you lack this skill.” The assistant should not infer demographic attributes or automate hiring decisions. Swedish requirements remain requirements even when an ad or answer is in English.

Ads can expire or change after download. The interface displays snapshot date, deadlines and original links and avoids promising current availability. Retrieved samples cannot establish Sweden-wide skill-frequency rankings. Semantic similarity thresholds are heuristic; they can admit irrelevant sources or suppress relevant ones, especially cross-language or in short queries.

CV text is held in the application session rather than persisted to disk/index. Users are asked to remove identifying details and explicitly enable CV transmission. OpenAI mode sends selected context and consented CV content to an external provider; response `store=False` does not mean zero retention under all provider policies. Local Ollama reduces external transmission for generation. Real CVs should not be used in shared evaluation artifacts.

PDF extraction/OCR occurs locally in memory; uploaded PDFs are not sent to the generation provider. OCR can misread names, skills, dates, columns and accented characters, so the review step is necessary. Native, scanned and mixed-PDF tests validate the extraction paths; they do not establish accuracy on every CV layout. Upload limits are 10 MB / 12 pages, and extracted text above 16,000 characters is shortened with an explicit warning.

## Conclusion to complete

Summarize measured retrieval and generation tradeoffs after reviewing the results. Describe observed failure cases, their practical implications and improvements supported by evidence, such as better labels, a reranker or more representative sampling.

## Contributions and AI use

Record actual work undertaken by each team member. The original plan assigns data/retrieval to Sim, generation to Brad, and UI/evaluation to Elvia; these are planned roles, not verified contributions.

Suggested AI-use declaration to adapt truthfully: “We used an AI coding assistant to help implement the ingestion, retrieval, generation, interface, evaluation tools and documentation. We reviewed and tested the code and manually verified experimental labels and claims. [Describe actual review, changes and final authorship.]” Do not claim manual review or experiments that have not happened.

## References

- Arbetsförmedlingen, [JobStream API](https://jobstream.api.jobtechdev.se/swagger.json).
- Wang et al., [Multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small).
- OpenAI, [Responses streaming documentation](https://developers.openai.com/api/docs/guides/streaming-responses).
- [Chroma documentation](https://docs.trychroma.com/).
