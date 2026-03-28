# Adaptation by Level: Words -> Chapters

Use this when rewriting English technical source into native-sounding Russian for Habr.

The rule is simple: **do not translate line by line**. Rebuild the text level by level.

Each level has different tools, different failure modes, and different quality signals. Word-level consistency is necessary but not sufficient. Phrase-level adaptation fixes calques. Sentence-level rewriting fixes flow. Paragraph-level restructuring fixes logic. Chapter-level adaptation fixes the reading experience.

## 1) Words

### What to do

- Normalize all single-term choices via `GLOSSARY.md`.
- Pick one preferred variant and keep it across the article.
- Use bilingual first mention only when it actually reduces ambiguity.

### What usually goes wrong

- The same concept appears under 3 names.
- English stays in place because the source used it.
- One paragraph is clean Russian, the next slips into mixed jargon.

### Examples

| Source concept | Bad inconsistency | Stable choice |
|---|---|---|
| model capacity | `капасити`, `мощность`, `ёмкость` | `ёмкость модели` |
| policy | `policy`, `augmentation policy`, `политика` | `политика аугментаций` or `augmentation policy` — pick one |
| label | `лейбл`, `метка` | `метка` |
| blur | `blur`, `блюр`, `размытие` | `размытие` in prose |
| feature | `фича`, `признак`, `фичер` | `признак` |
| overfitting | `оверфиттинг`, `переобучение` | `переобучение` |
| robustness | `робастность`, `устойчивость` | `устойчивость` |
| magnitude | `магнитуда`, `интенсивность`, `сила` | `интенсивность` or `сила` |
| transform | `трансформ`, `трансформация`, `преобразование` | `трансформация` in Albumentations context |

### Decision process per term

1. Check GLOSSARY.md — is there a ruling?
2. If not, apply the decision ladder: code? established? translatable?
3. Pick one form.
4. Search-and-replace the article for consistency.

## 2) Phrases and Collocations

### What to do

- Translate phrase meaning, not word sequence.
- Replace English collocations with stable Russian collocations.
- Use `COMMON-PHRASES.md` for recurring patterns.

### The core principle

English collocations carry meaning as a unit. Translating each word independently produces unnatural Russian. The Russian phrase should carry the same semantic payload, not mirror the English word order.

### Common phrase rewrites

| English phrase | Bad literal Russian | Better Russian |
|---|---|---|
| default outcome | дефолтный исход | типичный результат |
| edge case | edge case | редкий крайний случай / редкость |
| kitchen-sink policy | кухня-раковина policy | подход «суём всё подряд» |
| train hard, test easy | тренируй сильно, тестируй легко | тренируйся на сложном, тестируйся на лёгком |
| keep what helps | держите то, что помогает | оставляйте только то, что реально помогает |
| bridge the gap | навести мост через разрыв | сократить разрыв / закрыть разрыв |
| silently corrupt training | тихо портить тренировку | незаметно портить обучение |
| fire-and-forget decision | решение огонь-и-забыл | настроил и забыл |
| the model captures a slice of reality | модель захватывает слайс реальности | модель фиксирует лишь срез реальности |
| fails catastrophically | катастрофически ломается | сильно проседает / перестаёт работать |
| high-leverage tool | высокорычажный инструмент | один из самых эффективных инструментов |
| cheap to integrate | дешёвый для интеграции | легко встроить |

### Section-label rewrites

- `Failure mode:` -> `Когда вредит:`
- `What can go wrong:` -> `Где это ломается:`
- `Key takeaway:` -> `Главный вывод:`
- `Why each transform is there:` -> `Что делает каждая трансформация:`

## 3) Sentences

Russian technical prose usually wants:

- the main claim earlier,
- fewer nested clauses,
- more direct verbs,
- fewer passive constructions,
- less syntactic imitation of English.

### Sentence rewrite algorithm

1. Find the semantic nucleus — what is this sentence really saying?
2. Put the main claim near the start.
3. Move caveats and examples later.
4. Split if the English sentence carries two or more heavy clauses.
5. Replace abstract nouns with verbs where possible.
6. Check: would a Russian ML engineer say this aloud? If not, rewrite.

### Typical sentence problems

| Problem | Weak Russian | Better Russian |
|---|---|---|
| Passive overload | `метрика может быть ухудшена` | `метрика ухудшается` |
| Abstract noun pileup | `происходит улучшение устойчивости` | `модель становится устойчивее` |
| Literal English order | `Это происходит потому, что модель...` everywhere | sometimes start directly with `Модель...` |
| Too many caveats | 1 sentence with 4 commas and 2 dashes | split into 2 sentences |
| Throat-clearing | `Следует отметить, что...` | drop it, state the fact directly |
| Abstract noun + weak verb | `осуществляет обработку данных` | `обрабатывает данные` |

### Real example: sentence splitting and restructuring

**English (one long sentence):**

> The training set captures a specific slice of reality — particular lighting, particular cameras, particular weather, particular framing conventions — and the model learns to exploit those specifics rather than the semantic content that actually matters.

**Bad Russian (mirrors English structure):**

> Обучающий набор захватывает специфический слайс реальности — конкретное освещение, конкретные камеры, конкретную погоду, конкретные конвенции кадрирования — и модель учится эксплуатировать эти специфики, а не семантическое содержимое, которое реально важно.

**Good Russian (restructured, split, natural):**

> Датасет фиксирует лишь небольшой срез реальности: освещение, камеры, погоду и ракурсы — и модель опирается на детали съёмки, а не на семантику изображения.

What changed:
- `обучающий набор` → `датасет` (shorter, more natural in ML context)
- `захватывает специфический слайс` → `фиксирует лишь небольшой срез` (natural Russian verb + natural Russian noun)
- dash-separated list condensed
- `exploits those specifics` → `опирается на детали съёмки` (Russian verb, concrete action)
- `semantic content that actually matters` → `семантику изображения` (shorter, cleaner)

### Real example: passive to active

**English:**

> When label preservation fails, augmentation becomes label noise. The model receives contradictory supervision and performance degrades — often silently, because aggregate metrics can mask per-class damage.

**Bad Russian (passive, follows English word order):**

> Когда сохранение метки нарушается, аугментация становится шумом меток. Модель получает противоречивый обучающий сигнал, и производительность деградирует — часто тихо, потому что агрегированные метрики могут маскировать ущерб по классам.

**Good Russian (active, restructured):**

> Когда сохранение метки нарушается, аугментация превращает данные в шум. Модель получает противоречивый обучающий сигнал, и метрики ухудшаются — часто незаметно, потому что агрегированные метрики могут маскировать ущерб по отдельным классам.

What changed:
- `becomes label noise` → `превращает данные в шум` (active verb, clearer causation)
- `performance degrades` → `метрики ухудшаются` (concrete: what performance? metrics)
- `silently` → `незаметно` (natural Russian adverb)
- `per-class` → `по отдельным классам` (natural prepositional phrase)

### Real example: removing English scaffolding

**English:**

> This observation is the foundation of image augmentation: many transformations change the pixels of an image without changing what the image means. The technical term is that the label is invariant to these transformations.

**Good Russian:**

> На этом и строится идея аугментации: многие преобразования меняют пиксели, не меняя смысл изображения. Технический термин — метка инвариантна к этим преобразованиям.

What changed:
- `This observation is the foundation of` → `На этом и строится идея` (natural Russian construction)
- `without changing what the image means` → `не меняя смысл изображения` (participial phrase instead of relative clause)
- `The technical term is that` → `Технический термин —` (dash instead of verbose linking)

## 4) Paragraphs

The paragraph is the real unit of adaptation. English docs often rely on implicit transitions that Russian Habr prose should make explicit.

### Good paragraph shape

1. Why it matters (or the claim).
2. What happens technically (mechanism).
3. What the reader should do with it (practical implication).

### Keep paragraph roles explicit

Each paragraph should have a dominant role:

- claim,
- mechanism,
- consequence,
- practical action,
- failure mode.

If one paragraph tries to do all five, split it.

### What usually needs rewriting

- compress repeated English scaffolding,
- add one bridge sentence where the source assumes the inference is obvious,
- turn docs-style explanation into article-style argument.

### Real example: intro paragraph adaptation

**English intro (docs style):**

> A model trained on studio product photos fails catastrophically when users upload phone camera images. A medical classifier that achieves 95% accuracy in the development lab drops to 70% when deployed at a different hospital with different scanner hardware. A self-driving perception system trained on California summer data struggles in European winter conditions. A wildlife monitoring model that works perfectly on daytime footage collapses when the camera trap switches to infrared at dusk.

**Good Russian intro (Habr style):**

> Модель, обученная на студийных фотографиях, сильно проседает, когда пользователи загружают снимки с телефона. Медицинский классификатор с точностью 95% падает до 70% при развёртывании в соседней больнице с другим оборудованием. Система восприятия для автономного вождения, обученная на летних калифорнийских данных, теряется в зимних европейских условиях. Модель мониторинга дикой природы, отлично работающая днём, перестаёт работать, когда камера-ловушка переключается в инфракрасный режим на закате.

What changed:
- `fails catastrophically` → `сильно проседает` (natural Russian, not calque `катастрофически`)
- `different scanner hardware` → `другим оборудованием` (shortened, same meaning)
- `struggles in` → `теряется в` (natural Russian verb)
- `collapses when` → `перестаёт работать, когда` (natural, not `коллапсирует`)
- `self-driving perception system` → `Система восприятия для автономного вождения` (Russian word order)
- `camera trap` → `камера-ловушка` (standard Russian term)

### Real example: follow-up paragraph compression

**English (two paragraphs):**

> These are not rare edge cases. They are the default outcome when models memorize the narrow distribution of their training data instead of learning the underlying visual task. The training set captures a specific slice of reality — particular lighting, particular cameras, particular weather, particular framing conventions — and the model learns to exploit those specifics rather than the semantic content that actually matters.
>
> The primary solution is to collect data from the target distribution where the model will operate. There is no substitute for representative training data. But data collection is expensive, slow, and often incomplete — you cannot anticipate every deployment condition in advance.

**Good Russian (compressed, restructured):**

> Это не экзотика — модель просто запоминает узкое распределение обучающих данных вместо решения самой задачи. Датасет фиксирует лишь небольшой срез реальности: освещение, камеры, погоду и ракурсы — и модель опирается на детали съёмки, а не на семантику изображения.
>
> Лучшее решение — собрать данные из того распределения, в котором модель будет работать. У репрезентативных обучающих данных нет альтернатив. Но сбор данных — это дорого, долго и всегда неполно: невозможно предусмотреть все условия эксплуатации модели заранее.

What changed:
- `These are not rare edge cases. They are the default outcome...` → `Это не экзотика — модель просто запоминает...` (two sentences compressed into one, more direct)
- `The primary solution` → `Лучшее решение` (cleaner Russian)
- `There is no substitute for` → `У X нет альтернатив` (natural Russian construction)
- `expensive, slow, and often incomplete` → `дорого, долго и всегда неполно` (parallel Russian adjectives; `often` → `всегда` — deliberate strengthening for Habr impact)
- `you cannot anticipate every deployment condition in advance` → `невозможно предусмотреть все условия эксплуатации модели заранее` (impersonal, which is more natural in Russian)

## 5) Sections

Section titles should sound like Russian article subheads, not mirrored docs headers.

### Prefer meaning-first headings

| English source | Weak literal heading | Better Habr heading |
|---|---|---|
| What Is Image Augmentation? | Что такое аугментация изображений? | Аугментация изображений: от интуиции до продакшена |
| The Intuition: Transforms That Preserve Meaning | Интуиция: трансформации, которые сохраняют значение | Интуиция: трансформации, сохраняющие смысл |
| Why Augmentation Helps: Two Levels | Почему аугментация помогает: два уровня | Почему аугментация помогает: два слоя |
| The One Rule: Label Preservation | Одно правило: сохранение метки | Единственное правило: сохранение метки |
| Build Your First Policy | Постройте вашу первую policy | Первый пайплайн: базовая политика аугментаций |
| Prevent Silent Label Corruption | Предотвратите тихую порчу меток | Синхронизация таргетов: как не испортить метки незаметно |
| Expand the Policy Deliberately | Расширяйте policy обдуманно | Расширяйте политику обдуманно: семейства трансформаций |
| Tune Systematically | Настраивайте систематически | Настраивайте систематически: вероятность и интенсивность |
| Know the Failure Modes | Знайте режимы отказа | Типичные ошибки: лучше знать до production |
| Task-Specific and Targeted Augmentation | Специфичная для задачи аугментация | Аугментация под задачу |
| Evaluate With a Repeatable Protocol | Оценивайте с воспроизводимым протоколом | Оценка по воспроизводимому протоколу |
| Advanced: Why These Heuristics Work | Продвинуто: почему эти эвристики работают | Продвинутый раздел: почему эти эвристики работают |
| Production Reality: Operational Concerns | Production реальность | Production: эксплуатационные аспекты |

### Section heading patterns for Habr

Good Habr heading patterns:
- `[Topic]: [scope or benefit]` — `Аугментация изображений: от интуиции до продакшена`
- `[Action verb] [object] [qualifier]` — `Настраивайте систематически: вероятность и интенсивность`
- `[Topic]: [what reader gets]` — `Типичные ошибки: лучше знать до production`

Bad Habr heading patterns:
- Documentation-style slot names: `Overview`, `Introduction`, `Background`
- Abstract nouns alone: `Failure Modes`, `Considerations`
- Question-only headers that don't promise anything: `What Is X?`

### Sub-section ordering within a section

When the source is doc-shaped, Russian Habr flow is often better as:

1. problem / why the reader should care,
2. intuition / mechanism,
3. implementation / code,
4. failure modes / caveats,
5. practical action / summary.

Do not preserve doc order if it weakens the article.

## 6) Whole Chapters / Full Article Flow

For Habr, a chapter should not feel like "section 4 of documentation". It should feel like a coherent article segment with its own mini-arc.

### Recommended article rhythm

1. Hook:
   - concrete failure,
   - surprising metric drop,
   - deployment pain.
2. Scope:
   - what this article covers,
   - what it does not.
3. Roadmap:
   - short numbered list if the chapter is long.
4. Main build-up:
   - concept -> mechanism -> practice.
5. Pitfalls:
   - where the naive reader will break things.
6. Evaluation / verification:
   - how to check whether the advice actually helped.
7. Conclusion:
   - synthesis,
   - practical takeaway,
   - soft next step.

### Chapter-level adaptation rules

- Add a stronger intro than the doc source has.
- Remove navigation-only sections (`Where to Go Next`, `See also`, `Related pages`).
- Reframe docs navigation into standalone further reading or a compact ending.
- Ensure each large section answers one of: why, how, when it fails, how to verify.
- If the docs have a long `Where to Go Next` list, replace it with a concise closing paragraph.

### Real example: ending adaptation

**English ending (docs navigation):**

> ## Where to Go Next
> - Install Albumentations
> - Translations: Русский (Habr)
> - Learn Core Concepts
> - How to Pick Augmentations
> - Basic Usage Examples
> - Supported Targets by Transform
> - Explore Transforms Visually

**Good Russian ending (standalone Habr):**

> ## Заключение
>
> Аугментация изображений — один из самых эффективных инструментов в компьютерном зрении. [synthesis paragraph]
>
> Практический план действий:
> 1. Начните с in-distribution, сохраняющих метку трансформаций...
> [actionable numbered list]
>
> ---
> Albumentations — open-source библиотека для аугментации изображений.
>
> Если хотите поэкспериментировать с аугментациями на практике, можно попробовать библиотеку Albumentations или посмотреть, как работают отдельные трансформации в интерактивном инструменте Explore Transforms.

What changed:
- Navigation list → synthesis + actionable summary + soft CTA
- Multiple internal doc links → two external links only
- Docs-style "see also" → standalone conclusion that works without the docs site

### Real example: roadmap adaptation

**English roadmap:**

> This guide follows one practical story from first principles to production:
> 1. understand what augmentation is and why it works,
> 2. design a starter policy you can train with immediately,
> 3. avoid the failure modes that silently damage performance,
> 4. evaluate and iterate using a repeatable protocol,
> 5. then go deeper into theory and operational constraints.

**Good Russian roadmap:**

> В этой статье — путь от основ до production. Мы будем говорить об аугментации изображений, хотя многое будет применимо и к другим модальностям.
>
> План:
> 1. Понять, что такое аугментация, зачем она нужна и почему работает.
> 2. Собрать стартовый пайплайн, с которым сразу можно обучать.
> 3. Разобрать типичные ошибки, которые незаметно портят результат.
> 4. Оценить результаты по воспроизводимому протоколу.
> 5. Разобрать теорию и ограничения в production.

What changed:
- `This guide follows one practical story from first principles to production` → `В этой статье — путь от основ до production` (shorter, punchier)
- Added meta-comment about scope: `Мы будем говорить об аугментации изображений, хотя многое будет применимо и к другим модальностям` (sets expectations)
- Each list item starts with an infinitive verb (Russian convention for action plans)
- `understand what augmentation is and why it works` → `Понять, что такое аугментация, зачем она нужна и почему работает` (expanded for clarity — Russian needs more explicit structure)
- `design a starter policy you can train with immediately` → `Собрать стартовый пайплайн, с которым сразу можно обучать` (concrete verb + concrete noun)

## 7) Tone and Register for Habr

- Professional, direct, slightly conversational.
- Not bureaucratic, not academic-dead, not startup-bro slang.
- Use `вы` in guidance and imperative sections.
- Use impersonal style for general claims when it reads cleaner.
- Use `мы` sparingly, only for shared narrative ("we" = author + reader exploring together).

Good:

- `Начните с консервативной политики.`
- `Здесь модель обычно ломается.`
- `Это полезно, когда условия деплоя гуляют сильнее, чем обучающие данные.`
- `Не нужно перечислять все возможные варианты.`
- `Аугментация — мощный вспомогательный инструмент.`

Bad:

- `В рамках данного подхода осуществляется...`
- `Следует отметить, что...`
- `Данная секция рассматривает...`
- `В целях повышения качества...`
- `Предлагаемый метод позволяет...`
- `На основании вышеизложенного...`

### The `данный` test

If you wrote `данный` anywhere, replace it:
- `данный подход` → `этот подход`
- `данная трансформация` → `эта трансформация`
- `в данном случае` → `здесь` / `в этом случае`

`Данный` is a marker of translated, bureaucratic Russian. Native technical prose almost never uses it.

## 8) Micro-Style Rules

- Use `ё` consistently where standard (`ещё`, `ёмкость`, `всё`, `её`).
- Use Russian quotes `«...»` in prose for Russian quoted phrases.
- Keep English quotes only for exact terms and code contexts.
- Use a dash (`—`) when it genuinely improves flow, not as a decoration.
- Avoid overloading one sentence with parentheses, dashes, and commas at once.
- For emphasis, prefer bold over italic in Habr (italic is harder to read on screen).
- Use `\` for math: `×2`, not `x2`.

## 9) Final Structural Check

Before shipping a chapter, ask:

- At word level: are terms stable? Did I use `GLOSSARY.md`?
- At phrase level: did I translate collocations rather than individual words?
- At sentence level: does the syntax sound Russian? Could I read it aloud naturally?
- At paragraph level: does each paragraph have a job? Is the role clear?
- At section level: do headings promise something to the reader?
- At chapter level: does the whole piece read like Habr, not docs?
- At article level: would a Russian ML engineer share this on their team Slack?
