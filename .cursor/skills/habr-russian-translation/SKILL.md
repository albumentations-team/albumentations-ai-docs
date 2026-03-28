---
name: habr-russian-translation
description: Adapts English technical docs and blog posts into publication-ready Russian for Habr. Covers terminology choices, anti-anglicism policy, phrase/sentence/paragraph/chapter rewriting, Habr formatting, and QA. Use when translating or rewriting English technical content for Russian-speaking Habr readers.
---

# English -> Russian for Habr

Use this skill for **English -> Russian** Habr adaptation of technical material. This is not literal translation, not "just make it Russian", and not generic Russian copyediting. The target is native, editorially strong Russian technical prose that reads like a solid Habr post by a practicing ML/CV engineer.

## Use This Skill When

- The source is in English and the output must be in Russian.
- The result should be publishable on Habr, not merely understandable in Russian.
- The task includes translation, adaptation, restructuring, or editorial rewriting.
- The text is technical and the English source structure still matters.

Do **not** treat this as a generic "write Russian" skill for original Russian copy. Its main job is source-aware EN -> RU adaptation.

## What This Is NOT

- Not word-for-word translation. If the output mirrors the English sentence structure, it failed.
- Not "Russian plus scattered English terms." The text must sound like it was originally written in Russian.
- Not a docs page in Russian. Habr readers expect an article with a hook, a narrative, and a payoff.
- Not maximum Russification either. Established ML terms (аугментация, пайплайн, датасет, инференс) stay as they are.

## What Good Output Looks Like

- The text sounds originally written in Russian, not translated from English.
- English is kept only where it is canonical (code names, API), established (аугментация, пайплайн), or more precise.
- The article works as a standalone Habr post, not as a copied docs page.
- The chapter rhythm is adapted for readers: hook, mechanism, consequences, practice, failure modes, conclusion.
- A Russian ML engineer would read it and think "this is well-written," not "this is translated."

## Mandatory Workflow

1. **Read the full English source** and identify each section's role:
   - hook,
   - concept,
   - implementation,
   - pitfall,
   - evaluation,
   - conclusion.

2. **Extract the semantic core** before translating wording:
   - main claim,
   - mechanism,
   - practical consequence,
   - action for the reader.

3. **Build a translation ledger before drafting**:
   - English concept,
   - chosen Russian form,
   - whether English stays,
   - first-mention format,
   - article-wide consistency note.

4. **Normalize vocabulary** — word level:
   - single terms via `GLOSSARY.md`,
   - recurring collocations via `COMMON-PHRASES.md`.
   - Pick one term per concept and lock it for the whole article.

5. **Apply anti-anglicism policy** — kill bad English:
   - kill hybrid verbs (`заапплаить`, `матчить`, `хэндлить`, etc.),
   - replace weak transliterations (`боттлнек` → `узкое место`, `лейбл` → `метка`),
   - translate weak English nouns into proper Russian,
   - keep English only when it is established or canonical,
   - check false friends (`production` ≠ `производство`, `actual` ≠ `актуальный`).
   - Full rules: `ANTI-ANGLICISMS.md`.

6. **Rebuild the text by level** — do not translate line by line:
   - **Words**: normalize, stabilize, lock.
   - **Phrases**: translate collocations as units, not word sequences.
   - **Sentences**: move main claim forward, split long chains, reduce passive, use strong verbs.
   - **Paragraphs**: make the role explicit: claim → mechanism → consequence → practical takeaway.
   - **Sections**: rewrite headings for Habr, not docs. Headings should promise something.
   - **Whole chapters**: optimize for Habr flow, not for docs navigation order.
   - Full guide with real examples: `STRUCTURE-BY-LEVEL.md`.

7. **Run the full article workflow**:
   - Use `TRANSLATION-WORKFLOW.md`.
   - Separate source analysis from lexical normalization.
   - Separate translation from Habr packaging.
   - Do one final "does this still sound translated?" read-through.

8. **Adapt for Habr publishing** — the text must work as a Habr post:
   - Rewrite the title (don't mirror English mechanically).
   - Add an intro hook (concrete failure, not a definition).
   - Ban vague cliché hook subjects like "Most people...", "People often...", "Many teams...", "Usually...", unless immediately tied to a specific workflow or failure mode.
   - Convert admonitions to blockquotes.
   - Replace relative docs links with public URLs or remove them.
   - Translate image alt text.
   - Remove docs navigation sections.
   - Write a standalone conclusion with synthesis + practical takeaway.
   - Full rules: `HABR-FORMATTING.md`.

9. **Run the anti-anglicism linter**:
   - `python .cursor/skills/habr-russian-translation/scripts/term-lint.py <file-or-dir>`
   - Review all findings. Lint passing does not mean the prose is clean — manual review is still required.

10. **Run the final review checklist** from `QA-CHECKLIST.md`:
   - Language quality, terminology consistency, anti-anglicism, structural adaptation, Habr formatting, technical accuracy, typography, editorial quality.

## Non-Negotiable Rules

### Code and API names
- Keep transform/API/class names in English exactly as in code/docs.
- Keep code identifiers, parameters, metrics, and inline code unchanged.
- Never localize variable names or API calls.

### Prose
- Translate prose, captions, alt text, and prose comments around code.
- Use bilingual first mention only when it improves precision; drop the English after that.
- If English remains in prose after the first mention, there must be a reason.
- Prefer Russian verbs and Russian nouns over lazy English carry-over.
- Prefer everyday strong Russian over dictionary-valid but dead-sounding words. If a word feels like a calque even though it is technically Russian (`безадресно`, `волосные трещины` in the wrong context), rewrite it into something a practicing engineer would actually say.
- Do not mirror English sentence order or paragraph shape mechanically.
- Do not hide behind vague subjects in intros. Replace "люди часто", "многие команды", "обычно делают так" with a concrete scene, workflow, or failure.
- Do not end with fake-interest formulas like "интересно, что думают другие" if the real goal is to ask for feedback. Ask directly and concretely.
- Keep credibility attached to the material, not the author's ego. Prefer "текст основан на официальной документации" plus a short role marker like "от одного из создателей" or "от ключевого мейнтейнера", instead of several first-person status claims подряд.
- Avoid empty politeness endings like "буду признателен", if they add no meaning. Prefer a more human and specific ask: "будет интересно узнать о вашем опыте", "буду рад примерам, где это ломается", "напишите, если в вашей задаче это не работает".
- Kill `данный` — replace with `этот` or restructure.
- Kill `осуществляется`, `в рамках`, `следует отметить` and similar bureaucratic filler.
- Kill `коллапсировать` — use `ломаться`, `проседать`, `разваливаться`.
- In prose, prefer `горизонтальное отражение` / `вертикальное отражение` over `флип`, unless you are naming the exact transform/API token like `HorizontalFlip` or `VerticalFlip`.
- Watch for sentence patterns like "`X` утверждает:" after English `X claims`. In Russian prose this is often cleaner as `означает`, `задаёт предположение`, `предполагает`, or a direct declarative rewrite.
- Watch verb strength around labels. For English `destroys the label` / `corrupts the label`, prefer `ломает метку`, `портит метку`, `делает разметку шумной`, or `вносит шум в разметку` over emotionally stronger verbs like `уничтожает`.
- Prefer readable profession or role nouns over exotic niche borrowings. Example: `орнитолог` or `наблюдатель за птицами`, not `бёрдвотчер`.

### Formatting
- Replace relative docs links with absolute public URLs or plain text.
- Convert MkDocs admonitions to Habr-friendly blockquotes.
- Use `ё` consistently where standard (`ещё`, `ёмкость`, `всё`, `её`).
- Use Russian quotes `«...»` in Russian prose.
- Use em dash `—` with spaces for sentence breaks.
- Use multiplication sign `×`, not `x`.

## Rewrite Hierarchy

This is the core of the skill. Translation happens at five levels, bottom-up:

- **Word level**: choose one preferred term per concept and keep it stable. Use `GLOSSARY.md`.
- **Phrase level**: translate collocations and idioms, not isolated words. Use `COMMON-PHRASES.md`.
- **Sentence level**: move the main claim forward, split long chains, reduce passive voice, use strong verbs. Kill abstract noun + weak verb patterns.
- **Paragraph level**: make the role explicit: claim → mechanism → consequence → practical takeaway. Cut padding.
- **Chapter level**: optimize for Habr flow, not for docs navigation order. Rewrite headings. Add hooks. Remove docs navigation. Write standalone conclusion.

See `STRUCTURE-BY-LEVEL.md` for concrete before/after examples from real translations.

## Required Reference Files

- Terminology policy: [GLOSSARY.md](GLOSSARY.md)
- Common phrase translations: [COMMON-PHRASES.md](COMMON-PHRASES.md)
- Anti-anglicism rules: [ANTI-ANGLICISMS.md](ANTI-ANGLICISMS.md)
- Full article workflow: [TRANSLATION-WORKFLOW.md](TRANSLATION-WORKFLOW.md)
- Structural rewriting by level: [STRUCTURE-BY-LEVEL.md](STRUCTURE-BY-LEVEL.md)
- Habr formatting and publication adaptation: [HABR-FORMATTING.md](HABR-FORMATTING.md)
- Final review checklist: [QA-CHECKLIST.md](QA-CHECKLIST.md)
- Utility linter script: `scripts/term-lint.py`

## Final Self-Test

Before delivering the translation, ask yourself:

1. Would a Russian ML engineer write this sentence this way from scratch?
2. Did I keep English because it is canonical, or because I was too lazy to rewrite it?
3. Does this section still make sense if the reader never saw the original docs page?
4. Could I read the intro aloud at a meetup without cringing?
5. Did I check every term against the glossary?
6. Did I run the linter?
7. Would this article embarrass me if posted on a Russian ML Telegram channel?
