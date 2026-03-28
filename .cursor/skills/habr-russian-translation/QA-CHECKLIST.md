# EN -> RU Habr Localization QA Checklist

Run this before considering a translation final.

Passing this checklist means the text is not just translated, but editorially adapted for Habr.

## A) Language Quality

- [ ] The text reads like native Russian technical prose, not sentence-by-sentence translation.
- [ ] No paragraph obviously mirrors English syntax or pacing.
- [ ] Passive-heavy constructions were reduced where direct Russian is cleaner.
- [ ] No bureaucratic filler such as `данный`, `осуществляется`, `в рамках` abuse.
- [ ] The tone is professional and direct, not dead-academic and not startup-bro slang.
- [ ] Sentences are the right length: not mechanically mirroring long English sentences, not choppy either.
- [ ] Main claims appear near the beginning of sentences, not buried after caveats.
- [ ] The `данный` test: search for `данный` — replace every occurrence with `этот` or restructure.
- [ ] No `Следует отметить, что...` or similar throat-clearing.

## B) Terminology Consistency

- [ ] A small article-level translation ledger was created before or during drafting.
- [ ] Terms were normalized using `GLOSSARY.md`.
- [ ] Multi-word technical phrases were checked against `COMMON-PHRASES.md`.
- [ ] One term per concept is used across the whole article (search for variations).
- [ ] Transform/API/class names remain in canonical English.
- [ ] Code identifiers, metrics, and parameters were not localized.
- [ ] Bilingual first mention was used only where it genuinely helps.
- [ ] After first bilingual mention, the English was dropped (not repeated every time).
- [ ] No false friends: `production` ≠ `производство`, `actual` ≠ `актуальный`, `capacity` ≠ `вместимость`.
- [ ] Common leftover English nouns in prose (`baseline`, `policy`, `grayscale`, `pixel-level`, `spatial`, `workflow`, `takeaway`) were reviewed explicitly.

## C) Anti-Anglicism Hygiene

- [ ] No hybrid verb slang remains: `заапплаить`, `матчить`, `хэндлить`, `скейлить`, `имплементить`, `пофиксить`, `задеплоить`, `протюнить`, `перетрейнить`, etc.
- [ ] Weak transliterations were replaced with proper Russian where possible: `боттлнек` → `узкое место`, `лейбл` → `метка`, `фича` → `признак`, `капасити` → `ёмкость`, `робастность` → `устойчивость`, `перформанс` → `качество/метрики`.
- [ ] Long raw English noun clusters in prose were rewritten (no 3+ English nouns in a row in Russian text).
- [ ] Standalone raw English nouns left in prose were checked one by one and kept only if justified.
- [ ] Replacements from `ANTI-ANGLICISMS.md` were applied.
- [ ] `term-lint.py` was run on the final text and all findings were reviewed.
- [ ] Office-speak calques were removed: `является инструментом` → `помогает`, `осуществляет обработку` → `обрабатывает`.
- [ ] `коллапсировать` → `ломаться` / `проседать`.
- [ ] The text does not pass lint while still sounding translated (lint is necessary but not sufficient).

## D) Structural Adaptation by Level

### Word level
- [ ] Normalized and stabilized (one term per concept).
- [ ] No inconsistent switching between Russian and English for the same term.

### Phrase level
- [ ] Idioms and collocations were adapted, not calqued.
- [ ] Section labels were rewritten for Habr (`Failure mode:` → `Когда вредит:`).
- [ ] Common English phrases were checked against `COMMON-PHRASES.md`.

### Sentence level
- [ ] Long English chains were split or rebuilt for Russian flow.
- [ ] Passive voice was reduced to natural Russian levels.
- [ ] Abstract noun + weak verb patterns replaced with strong verbs.
- [ ] Main claims moved toward the beginning of sentences.

### Paragraph level
- [ ] Each paragraph has a clear role and a visible logical arc.
- [ ] No paragraph is pure padding or transition-only.
- [ ] Docs-style explanation paragraphs were turned into article-style arguments.
- [ ] Bridge sentences were added where the source assumed obvious inferences.

### Section level
- [ ] Headers were rewritten for Russian reading flow.
- [ ] Headers promise something to the reader, not just label a topic.
- [ ] No abstract docs-slot headers remain (`Overview`, `Background`, `Introduction`).

### Chapter / article level
- [ ] The overall flow feels like Habr, not docs.
- [ ] Navigation-only sections were removed or reframed.
- [ ] The article works as a standalone piece.
- [ ] Each large section clearly answers one of: why, how, where it breaks, how to verify.

## E) Habr Publishing Adaptation

### Links
- [ ] Relative docs links were replaced with absolute public URLs or removed.
- [ ] No internal docs cross-references that Habr readers cannot reach.
- [ ] Transform names linked to Explore on first mention only, not every mention.
- [ ] No bare URLs in prose.

### Formatting
- [ ] MkDocs admonitions were converted to plain blockquotes (`> Важно. ...`).
- [ ] Image alt text was translated into Russian.
- [ ] Image title attributes were removed (`![alt](url "title")` → `![alt](url)`).
- [ ] Tables render correctly with translated column headers.
- [ ] Code blocks have language tags.
- [ ] No docs-only widgets, tabs, or collapsibles remain.

### Structure
- [ ] The intro has a hook (concrete failure/pain), not just a definition.
- [ ] The intro contains a clear scope sentence.
- [ ] Long articles have a short roadmap.
- [ ] The conclusion was rewritten for standalone Habr reading.
- [ ] The conclusion has: synthesis, practical takeaway, soft next step.
- [ ] Docs-only navigation sections (`Where to Go Next`) were removed or reframed.

## F) Technical Accuracy

- [ ] Nothing important was simplified into inaccuracy.
- [ ] Domain examples still match the original technical intent.
- [ ] Claims, caveats, and constraints from the source are preserved.
- [ ] Any local reordering did not break logical dependencies.
- [ ] Code examples are identical to the source (not modified during translation).
- [ ] Parameter values, metric names, and numbers are unchanged.
- [ ] Mathematical notation is correct and consistent.

## G) Typography and Markdown

- [ ] `ё` consistency was checked (`ещё`, `ёмкость`, `всё`, `её`, `объём`).
- [ ] Russian quotes `«...»` are used in prose where appropriate.
- [ ] Em dash `—` is used with spaces for sentence breaks.
- [ ] Hyphen `-` is used only for compound words.
- [ ] Multiplication sign `×` is used, not `x`.
- [ ] Code formatting is untouched.
- [ ] Lists render correctly in plain Markdown.
- [ ] Tables still read cleanly after translation.
- [ ] Blockquotes, links, and images are Habr-compatible.
- [ ] No double spaces or trailing whitespace.
- [ ] Consistent heading levels (no skipped levels).

## H) Final Editorial Pass

- [ ] Repetition that was useful in English docs was compressed where Russian became too heavy.
- [ ] No section is just "there because the docs had it" — every section earns its place.
- [ ] Each major section answers one of: `why`, `how`, `when it fails`, `how to verify`.
- [ ] The text keeps high information density without becoming compressed and unreadable.
- [ ] The ending gives the reader a concrete takeaway or next step.
- [ ] Read the first sentence of every section sequentially — do they tell a coherent story?
- [ ] Search for Latin tokens outside code, URLs, and canonical names: are the leftovers all justified?
- [ ] The whole article would not embarrass you if shared on a Russian ML engineers' Telegram channel.

## I) Pre-Publication Checklist

- [ ] `term-lint.py` passes with no unreviewed findings.
- [ ] Article was read through once without stopping to edit (final flow check).
- [ ] Habr tags were chosen (e.g., `Машинное обучение`, `Computer Vision`, `Обработка изображений`).
- [ ] Hub was chosen (e.g., `Машинное обучение`, `Обработка изображений`).
- [ ] Cover image is set (if Habr requires one).
