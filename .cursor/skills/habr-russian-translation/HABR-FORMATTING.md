# Habr Formatting and Publication Adaptation

This file covers the jump from **translated text** to **publishable Habr article**.

The goal is not merely Markdown compatibility. The goal is that the text behaves like a Habr post when read in the feed, in the body, and at the ending. A reader on Habr expects a coherent article, not a translated docs page.

## 1) Markdown Compatibility Essentials

Habr uses its own Markdown flavor. Some common docs features do not work.

### What works
- Standard headings (`##`, `###`)
- Bold (`**text**`) and italic (`*text*`)
- Fenced code blocks with language tags (```python)
- Pipe-syntax tables
- Blockquotes (`>`)
- Ordered and unordered lists
- Images (`![alt](url)`)
- Links (`[text](url)`)
- Inline code (`` `code` ``)

### What does not work or needs adaptation
- MkDocs admonitions (`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`) — convert to plain blockquotes
- Tabs, collapsibles, docs-only widgets — remove or inline
- Generated navigation — remove
- Title attributes on images (`![alt](url "title")`) — remove the title part
- Nested blockquotes — avoid
- HTML in Markdown — Habr supports some HTML but prefer pure Markdown

### Admonition conversion table

| Source style | Habr style |
|---|---|
| `> [!NOTE]` | `> Примечание. ...` |
| `> [!TIP]` | `> На практике. ...` |
| `> [!IMPORTANT]` | `> Важно. ...` |
| `> [!WARNING]` | `> Осторожно. ...` |

Keep the blockquote short. If the note becomes a full paragraph, make it normal body text instead.

**Example conversion:**

Source:
```
> [!IMPORTANT]
> The question is not "does this image look realistic?" but "is the label still obviously correct?"
```

Habr:
```
> Важно. Вопрос не в том, «выглядит ли это изображение реалистично?», а в том, «остаётся ли метка однозначной после трансформации?»
```

## 2) Titles and Section Headers

### Article title

Do not mirror the English title mechanically. The title is the most important line for Habr — it determines whether the reader opens the article.

Prefer:

- a strong topic noun,
- a value promise,
- a concrete scope,
- Habr rhythm.

Good pattern:

- `Тема: масштаб / ценность / путь`

Examples:

- `Аугментация изображений: от интуиции до продакшена`
- `Как подбирать аугментации: гипотезы, протокол и метрики`
- `Синхронизация таргетов: как не испортить метки незаметно`

Bad patterns:

- `Что такое аугментация изображений?` (docs FAQ, not article)
- `Аугментация изображений` (no hook, no promise)
- `Введение в аугментацию` (generic, boring)
- `Image Augmentation: A Comprehensive Guide` (English title not adapted)

### Section headers

- Make them meaning-first, not word-for-word.
- Prefer concrete reader-facing headers over abstract docs headers.
- If the English heading sounds like a documentation slot, rewrite it into a chapter claim.
- Headers should be scannable — a reader who only reads headers should understand the article structure.

| English heading | Bad translation | Better Habr heading |
|---|---|---|
| Why Augmentation Helps | Почему аугментация помогает | Почему аугментация помогает: два слоя |
| Failure Modes | Режимы отказа | Типичные ошибки: лучше знать до production |
| Build Your First Policy | Постройте вашу первую policy | Первый пайплайн: базовая политика аугментаций |
| Task-Specific Augmentation | Специфичная для задачи аугментация | Аугментация под задачу |
| Precision: target specific weaknesses | Точность: целевые слабости | Точечная настройка: работа со слабыми местами |

## 3) Intro Adaptation

The introduction must work for a Habr reader who has not opened the docs and may leave after 20 seconds.

### Required intro elements

1. A concrete failure or pain point in the first 2-3 sentences.
2. A clear scope sentence: what the article explains, what the reader will get.
3. A short roadmap if the article is long.

### What not to do

- Do not open with generic theory.
- Do not spend the first paragraph defining terms.
- Do not sound like docs onboarding copy.
- Do not open with `В данной статье рассматривается...`

### Intro structure pattern

```
[2-4 concrete failure/pain examples]

[One bridging sentence: "this is not exotic — it's the default"]

[One sentence: what the solution is]

[One sentence: what this article covers]

[Numbered roadmap if >5 sections]
```

### Real example

**English intro:**

> A model trained on studio product photos fails catastrophically when users upload phone camera images. [more examples...] These are not rare edge cases. They are the default outcome... The primary solution is to collect data from the target distribution... Image augmentation is the complementary tool... This guide follows one practical story from first principles to production:

**Good Habr intro:**

> Модель, обученная на студийных фотографиях, сильно проседает, когда пользователи загружают снимки с телефона. [more examples...]
>
> Это не экзотика — модель просто запоминает узкое распределение обучающих данных вместо решения самой задачи. [bridging]
>
> Лучшее решение — собрать данные из того распределения, в котором модель будет работать. [solution]
>
> Аугментация изображений — мощный вспомогательный инструмент, помогающий уменьшить проблему расхождения train и test распределений. [what augmentation does]
>
> В этой статье — путь от основ до production. [scope]
>
> План: [roadmap]

## 4) Body Adaptation

### Keep article momentum

- Alternate explanation with practical consequences.
- Let long theory sections periodically pay off in engineering terms.
- After dense conceptual blocks, add a short grounding sentence:
  - `На практике это означает...`
  - `Для пайплайна это важно потому, что...`
  - `Отсюда простое практическое правило:...`

### Replace docs-only navigation logic

Bad docs carry-over:

- "See the next chapter for..."
- "This page covers..."
- "Where to go next"
- "For more details, see..."
- "As described in the previous section..."

Better Habr adaptation:

- inline public links where they genuinely help
- short `если хотите углубиться` transitions
- a compact further-reading block at the end
- self-contained explanations that don't depend on another page

### Body paragraph density

Habr readers are engineers. They can handle dense paragraphs but need payoff. Rules:

- No paragraph should be pure padding.
- Every paragraph should add information, not just transition.
- If a paragraph exists only because the docs had it, cut it or merge it.
- Dense theory is fine if it leads to a practical implication within 2-3 paragraphs.

## 5) Links

### Link strategy

- Replace relative project links with absolute public URLs.
- If there is no good public target, remove the link and keep plain text.
- Keep external transform links to Explore when useful:
  - `https://explore.albumentations.ai/transform/<TransformName>`
- Avoid turning every mention into a link. Habr reads better with selective linking.
- Link transform names on first mention only, not every occurrence.
- Do not link internal docs pages — Habr readers cannot reach them.

### Link formatting

- Use descriptive link text, not bare URLs.
- Do not use "click here" or "see here" link text.
- Good: `[Explore Transforms](https://explore.albumentations.ai)`
- Bad: `[здесь](https://explore.albumentations.ai)` / bare URL in text

### What to do with docs cross-references

| Source pattern | Habr adaptation |
|---|---|
| `[Choosing Augmentations](../3-basic-usage/choosing-augmentations.md)` | Remove if no public URL; mention topic inline |
| `[Supported Targets by Transform](../reference/supported-targets-by-transform.md)` | Link to public docs URL or describe what to check |
| `see [Test-Time Augmentation](../4-advanced-guides/test-time-augmentation.md)` | Keep if public URL exists, otherwise inline the key info |
| `[Install Albumentations](./installation.md)` | Remove or replace with `pip install albumentations` |

## 6) Images

- Translate alt text into Russian.
- Remove title attributes from image syntax (`![alt](url "title")` → `![alt](url)`).
- Keep technically meaningful images, remove decorative duplicates.
- Prefer one image that teaches something over three repetitive screenshots.
- For final Habr publication, upload images to Habr CDN.
- In repo text, relative image paths are fine for drafting.

### Image alt text translation

| English alt | Russian alt |
|---|---|
| `One image, many augmentations` | `Одно изображение — множество аугментаций` |
| `Parrot label preservation under safe transforms` | `Сохранение метки при безопасных трансформациях` |
| `Training distribution widened by augmentation` | `Расширение обучающего распределения аугментацией` |
| `Mask and bbox synchronization` | `Синхронизация маски и bbox при трансформациях` |
| `Realistic vs over-augmented policy` | `Реалистичная vs переаугментированная политика` |

## 7) Code Blocks

- Keep code executable and close to source.
- Translate inline comments only when that improves readability for a Russian reader.
- Never localize variable names, parameters, class names, transform names, or API calls.
- If the surrounding prose already explains the code, do not over-comment the snippet.
- Keep the language tag (```python).

### Comment translation examples

**English code comment:**
```python
# Eyes: left (36-41) ↔ right (42-47)
```

**Russian code comment:**
```python
# Глаза: левый (36-41) ↔ правый (42-47)
```

**Do NOT translate:**
```python
# Do not translate this:
transform = A.Compose([...])  # Keep API calls untouched
```

## 8) Mixed Prose Around Code

- In narrative text, Russian should dominate.
- In code and canonical API names, English remains canonical.
- Use bilingual first mention only when ambiguity exists.
- Do not let prose degenerate into English noun clusters just because the nearby code is in English.

### Pattern: explaining code parameters in prose

Good:
> RandomResizedCrop вносит вариацию масштаба и кропа, не теряя смысла изображения.

Bad:
> RandomResizedCrop introduces scale и framing variation while preserving enough semantic content.

The prose around code should be Russian. The code itself stays English.

## 9) Tables

- Tables translate well into Habr Markdown.
- Translate column headers into Russian.
- Keep code/API names in English in table cells.
- Adjust column widths by keeping Russian text concise.

### Example table adaptation

**English:**

| Task | Input components | Albumentations targets |
|---|---|---|
| Classification | image | `image` |
| Object detection | image + boxes | `image`, `bboxes` |

**Russian:**

| Задача | Входные компоненты | Таргеты в Albumentations |
|---|---|---|
| **Классификация** | изображение | `image` |
| **Детекция объектов** | изображение + рамки | `image`, `bboxes` |

## 10) Math Notation

- Use multiplication sign `×` (Unicode ×), not `x` or `*`.
- Use proper subscripts and superscripts when Habr Markdown supports them.
- For inline math, use `$...$` if Habr renders it, otherwise use plain text with Unicode symbols.
- Keep formulas simple — Habr rendering is limited compared to full LaTeX.

Example: `2 × 31 × 5 = 310×` not `2 * 31 * 5 = 310x`

## 11) Ending and Conclusion

Do not copy docs navigation or a bare "next steps" list.

The ending should contain:

1. a short synthesis (1-2 paragraphs),
2. one practical takeaway (numbered actionable list),
3. a soft next step (not a hard product pitch),
4. optional public links for deeper reading.

### Good ending shape

- `Заключение` — synthesis paragraph
- `Практический план действий:` — numbered actionable list
- Separator (`---`)
- Soft CTA: mention the library and one interactive tool

### Bad ending shape

- docs table of contents,
- navigation-only bullets,
- hard product pitch,
- `Спасибо за внимание!` (unnecessary on Habr),
- `Подписывайтесь на канал` without context.

### Real ending example

```markdown
## Заключение

Аугментация изображений — один из самых эффективных инструментов в компьютерном зрении. [synthesis]

Практический план действий:
1. Начните с in-distribution, сохраняющих метку трансформаций...
2. Измерьте относительно baseline без аугментации.
3. [more items]

---
Albumentations — open-source библиотека для аугментации изображений.

Если хотите поэкспериментировать с аугментациями на практике, можно попробовать библиотеку Albumentations или посмотреть, как работают отдельные трансформации в интерактивном инструменте Explore Transforms.
```

## 12) Recommended Final Article Layout

1. Hook: production pain or a concrete failure.
2. Why the naive approach fails.
3. Core concept and intuition.
4. Practical baseline (code).
5. Failure modes and caveats.
6. Task-specific advice.
7. Validation or evaluation protocol.
8. Advanced theory if the article needs it.
9. Production / operational concerns.
10. Conclusion with soft CTA.

## 13) Typography Checklist

- [ ] `ё` used consistently (`ещё`, `ёмкость`, `всё`, `её`, `объём`)
- [ ] Russian quotes `«...»` used for Russian phrases in prose
- [ ] English quotes `"..."` kept only for code/API contexts
- [ ] Em dash `—` used for sentence breaks (with spaces: `слово — слово`)
- [ ] Hyphen `-` used only for compound words (`camera-ловушка`, `in-distribution`)
- [ ] Multiplication sign `×` used, not `x`
- [ ] No double spaces
- [ ] No trailing whitespace
- [ ] Consistent heading levels (no skipped levels)
