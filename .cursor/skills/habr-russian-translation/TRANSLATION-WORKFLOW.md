# EN -> RU Habr Translation Workflow

Use this file when the task is not "write Russian text", but specifically:

- take an English technical source,
- preserve its technical meaning,
- rebuild it as native Russian prose,
- publish it as a standalone Habr article.

This workflow is intentionally article-centric. It assumes you are translating and adapting, not just editing isolated Russian sentences.

## 1) Start with the source, not the words

Before touching sentence-level wording, extract the article structure from the English source.

For each section, write down:

| Section | Role in source | What the reader gets |
|---|---|---|
| Intro | hook + problem statement | why this matters |
| Concept section | definition + intuition | mental model |
| Practical section | pipeline / code / defaults | what to do |
| Failure section | caveats | what not to break |
| Evaluation section | protocol | how to verify |
| Conclusion | synthesis | what to remember |

If you cannot explain a section's role in one short line, you are not ready to translate it well.

## 2) Build a translation ledger before drafting

Do not improvise terminology paragraph by paragraph. Create a small ledger for the article first.

Recommended columns:

| English concept | Chosen Russian form | Keep English? | First mention format | Notes |
|---|---|---|---|---|
| augmentation | аугментация | no | plain Russian | established term |
| grayscale | оттенки серого | first mention only | `grayscale (оттенки серого)` | then Russian only |
| policy | политика аугментаций | usually no | plain Russian | avoid bare `policy` |
| label | метка | no | plain Russian | avoid `лейбл` |
| CoarseDropout | CoarseDropout | yes | exact English | code / API name |

Rules:

- One concept -> one preferred form.
- If English stays, there must be a reason: code token, canonical name, or entrenched ML term.
- If a bilingual first mention is needed, use it once and decay to the shorter Russian form after that.

## 3) Rewrite bottom-up by level

Do not translate line by line. Rewrite in this order:

1. **Words**: normalize terms and kill weak anglicisms.
2. **Phrases**: translate collocations as units.
3. **Sentences**: move the claim forward, split overloaded English syntax.
4. **Paragraphs**: give each paragraph one job.
5. **Sections**: rewrite headers and order for Habr readability.
6. **Whole article**: make the piece standalone.

Use:

- `GLOSSARY.md` for word-level decisions,
- `COMMON-PHRASES.md` for collocations,
- `ANTI-ANGLICISMS.md` for bad carry-over,
- `STRUCTURE-BY-LEVEL.md` for rewrite examples.

## 4) Word-level rules

At word level, solve consistency first.

Checklist:

- Decide whether the term is:
  - exact English code/API,
  - established Russian ML term,
  - normal Russian word.
- Remove lazy transliterations if Russian is cleaner.
- Normalize one variant across the article.

Typical wins:

- `label` -> `метка`
- `feature` -> `признак`
- `bottleneck` -> `узкое место`
- `robustness` -> `устойчивость`
- `capacity` -> `ёмкость`
- `throughput` -> `пропускная способность`

Typical keep-as-is cases:

- `CoarseDropout`
- `HorizontalFlip`
- `IoU`
- `mAP`
- `GPU`
- `SimCLR`

## 5) Phrase-level rules

English phrases often fail even when each individual word is translated correctly.

Translate the phrase meaning, not the surface form.

Examples:

| English | Bad literal rendering | Better Russian |
|---|---|---|
| default outcome | дефолтный исход | типичный результат |
| bridge the gap | построить мост | сократить разрыв |
| fails catastrophically | катастрофически ломается | сильно проседает / перестаёт работать |
| kitchen-sink policy | kitchen-sink policy | подход «суём всё подряд» |
| high-leverage tool | высокорычажный инструмент | один из самых эффективных инструментов |

If the phrase sounds like translated LinkedIn English, rewrite harder.

## 6) Sentence-level rules

English technical prose tolerates:

- longer chains,
- more passive voice,
- more caveats before the claim,
- more abstract noun + weak verb constructions.

Russian Habr prose prefers:

- claim first,
- concrete verb,
- cleaner syntax,
- fewer stacked clauses.

Rewrite algorithm:

1. Identify the semantic nucleus.
2. Put the main claim near the start.
3. Move caveats later.
4. Split long English chains when Russian gets heavy.
5. Replace abstract nouns with verbs.

Quick examples:

- `This is not a rare edge case.` -> `Это не редкость.` / `Это не экзотика.`
- `Augmentation is a complementary tool that helps bridge the gap.` -> `Аугментация помогает сократить этот разрыв.`
- `Performance degrades silently.` -> `Метрики ухудшаются незаметно.`

## 7) Paragraph-level rules

A paragraph should not be a pile of translated sentences. It needs a visible role.

Good paragraph roles:

- claim,
- mechanism,
- consequence,
- practical takeaway,
- failure mode.

Recommended shape:

1. Why the reader should care.
2. What happens technically.
3. Why it matters in practice.

If a paragraph only exists because the English source had a transition there, merge or cut it.

## 8) Section-level and chapter-level rules

English docs structure and Habr article structure are not the same.

For Habr:

- headings should promise value,
- sections should read like article chapters, not doc slots,
- the article must work without the reader opening the original docs page.

Default chapter rhythm:

1. Concrete failure or pain.
2. Explanation of why it happens.
3. Practical baseline or pattern.
4. Failure modes.
5. Verification protocol.
6. Conclusion with takeaway.

Rewrite headings when needed:

- `Failure Modes` -> `Типичные ошибки: где это ломается`
- `Build Your First Policy` -> `Первый пайплайн: базовая политика аугментаций`
- `Production Reality` -> `Production: эксплуатационные аспекты`

## 9) Habr packaging pass

After the translation is semantically stable, do a separate Habr pass.

Required:

- rewrite the title for click-worthiness and specificity,
- open with a concrete pain point, not a definition,
- replace doc-only cross-links,
- convert admonitions to blockquotes,
- translate alt text,
- remove navigation-only sections,
- write a standalone conclusion.

The final article should not feel like "docs page in Russian". It should feel like "strong Russian technical article that happens to be based on docs".

## 10) The anti-anglicism pass is separate

Do not assume good translation automatically kills anglicisms.

Run a dedicated pass for:

- hybrid verbs,
- raw English noun chains,
- weak transliterations,
- office-speak calques,
- English terms left in place only because the source had them.

Use:

- `ANTI-ANGLICISMS.md`
- `python .cursor/skills/habr-russian-translation/scripts/term-lint.py <file-or-dir>`

Important:

- Lint is only first-pass hygiene.
- Passing lint does not mean the prose is good.

## 11) The final read should be monolingual in effect

Even if the article keeps some English terms, the reading experience should still feel Russian.

Test:

- Read the first sentence of every section in order. Does it tell a coherent story?
- Read one dense paragraph aloud. Does it sound like something a Russian ML engineer would actually say?
- Search for leftover Latin tokens outside code, URLs, transform names, metrics, and known acronyms.
- Check whether any bilingual first mention was repeated three pages later for no reason.

If the answer is "this still sounds translated", go back up one level and rewrite structurally. Do not just swap a few words.

## 12) Deliverable standard

A good EN -> RU Habr adaptation:

- preserves the technical claims,
- sounds originally written in Russian,
- keeps only justified English,
- has stable terminology,
- has Habr-native title, intro, and ending,
- does not embarrass you in front of Russian ML/CV practitioners.
