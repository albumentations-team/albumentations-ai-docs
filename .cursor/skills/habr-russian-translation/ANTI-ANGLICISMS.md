# Anti-Anglicism Rules for Habr Russian

Goal: keep the text technically sharp, but remove the low-grade hybrid English/Russian sludge that makes Habr prose feel translated or lazy.

This file is not anti-English. It is anti-bad-English-in-Russian.

The core distinction: **established ML/CV terms** (аугментация, пайплайн, датасет, инференс) are fine. **Lazy carry-over** of English words that have clean Russian equivalents is not.

## 1) Decision Rule

Before keeping an English term, ask:

1. Is it a canonical name from code, API, a method, or a metric?
2. Is it a stable Russian ML term that practitioners actually use?
3. If I replace it with Russian, do I lose precision?

If the answer is "no" to all three, rewrite it into Russian.

Fourth question for the truly ambiguous cases: **Would a senior Russian ML engineer at Yandex, VK, or Sber say this aloud in a technical discussion?** If no, rewrite.

## 2) What Is Acceptable

- Established ML/CV nouns in Russian practice: `аугментация`, `пайплайн`, `датасет`, `инференс`, `чекпоинт`, `бэкбон`, `воркер`, `анкор`, `батч`.
- Canonical English transform/API/class names and code identifiers.
- Russian verb + accepted technical noun:
  - `собрать пайплайн`
  - `применить dropout`
  - `настроить learning rate`
  - `запустить валидацию`
  - `прогнать абляцию`
- Bilingual first mention when it prevents ambiguity:
  - `grayscale (оттенки серого)`
  - `augmentation policy (политика аугментаций)`
  - `pixel-level (пиксельные) трансформации`
- Hybrid adjectives from established terms:
  - `аугментационный пайплайн`
  - `регуляризационное давление`
  - `классоспецифичная аугментация`

## 3) What Is Not Acceptable

### Raw English leftovers that stayed only because the source had them

This is subtler than hybrid verbs, but still bad.

Typical pattern:

- the translator understands the English term,
- leaves it in prose "for now",
- never comes back,
- the final article contains scattered raw English that no longer serves a purpose.

Examples that usually need review:

| Raw English in prose | Usually better | Notes |
|---|---|---|
| `baseline` | `базовая точка отсчёта`, `baseline без аугментации` | choose one stable form |
| `policy` | `политика аугментаций`, `пайплайн` | bare `policy` is weak |
| `grayscale` | `оттенки серого` after first mention | bilingual once is enough |
| `pixel-level` | `пиксельные` | after first mention |
| `spatial` | `пространственные` | after first mention |
| `failure mode` | `когда вредит`, `режим отказа` | rewrite by context |
| `workflow` | `процесс`, `порядок работы` | keep English only for actual product/tool naming |
| `takeaway` | `главный вывод` | not `тейкауэй` |

Rule:

- English may stay on first mention if it reduces ambiguity.
- After that, English must either decay into Russian or remain only where it is canonical.

If you see the same raw English noun five pages later with no explanatory value, that is not precision. It is inertia.

### Hybrid English verbs with Russian endings

This is the worst category. Kill on sight.

| Bad form | What it signals | Replace with |
|---|---|---|
| `заапплаить` | lazy verb | `применить` |
| `матчить` | lazy verb | `соответствовать` / `совпадать` |
| `хэндлить` | lazy verb | `обрабатывать` |
| `скейлить` | lazy verb | `масштабировать` |
| `имплементить` | lazy verb | `реализовать` |
| `имплементировать` | heavy calque | `реализовать` / `внедрить` |
| `контрибьютить` | lazy verb | `вносить вклад` |
| `сетапить` | lazy verb | `настроить` / `собрать` |
| `пофиксить` | lazy verb | `исправить` |
| `фиксануть` | slang verb | `исправить` |
| `зааблейтить` | lazy verb | `провести абляцию` / `проверить абляцией` |
| `зарегуляризить` | ugly hybrid | `добавить регуляризацию` |
| `заинференсить` | ugly hybrid | `прогнать инференс` / `получить предсказание` |
| `отдебажить` | ugly hybrid | `отладить` |
| `задеплоить` | ugly hybrid | `развернуть` / `выкатить` |
| `проаугментировать` | ugly hybrid | `применить аугментацию` |
| `залоггировать` | ugly hybrid | `записать в лог` |
| `перетрейнить` | ugly hybrid | `переобучить` |
| `переаугментировать` | ugly hybrid | `переусилить аугментацию` |

### Weak transliterations when good Russian exists

| Bad transliteration | Replace with | Why |
|---|---|---|
| `боттлнек` | `узкое место` | Perfect Russian equivalent |
| `перформанс` | `качество` / `метрики` / `производительность` | Context-dependent but always has a Russian equivalent |
| `лейбл` | `метка` | Standard Russian ML term |
| `фича` | `признак` / `возможность` | Context-dependent |
| `робастность` | `устойчивость` | Clean Russian |
| `капасити` | `ёмкость` | Clean Russian |
| `трейд-офф` | `компромисс` / `баланс` | Clean Russian |
| `юзкейс` | `сценарий использования` / `применение` | Clean Russian |
| `сетап` | `настройка` / `конфигурация` | Clean Russian |
| `оверхед` | `накладные расходы` | Clean Russian |
| `тюнить` | `настраивать` / `подбирать` | Clean Russian |
| `рисёрч` | `исследование` | Clean Russian |
| `коллапсировать` | `разваливаться` / `ломаться` / `проседать` | See false friends |
| `аутпут` | `выход` / `результат` | Clean Russian |
| `инпут` | `вход` / `входные данные` | Clean Russian |
| `стейт` | `состояние` | Clean Russian |
| `конвертить` | `преобразовать` / `перевести` | Clean Russian |
| `чекнуть` | `проверить` | Clean Russian |

### Office-speak calques from English

These come from translating English management prose structure:

| Bad calque | Better Russian |
|---|---|
| `данный подход является...` | `этот подход работает как...` |
| `это имеет смысл только если...` | `это полезно, только когда...` |
| `в рамках данного исследования` | `в этой работе` / `здесь` |
| `осуществить процесс обучения` | `обучить` |
| `произвести валидацию` | `провалидировать` / `проверить` |
| `является инструментом` | `работает как инструмент` / `помогает` |
| `данная секция рассматривает` | `в этом разделе разберём` |
| `это может иметь эффект на` | `это влияет на` |
| `следует отметить, что` | drop it or `важно что` |

### Long chains of raw English nouns in Russian prose

If three or more English nouns sit in a row in Russian text, something went wrong.

| Bad | Better |
|---|---|
| `training loss plateaus unusually high` | `training loss выходит на необычно высокое плато` |
| `production deployment conditions` | `условия деплоя в production` |
| `augmentation policy failure modes` | `ошибки политики аугментаций` |
| `test-time augmentation transforms` | `TTA-трансформации` |

## 4) Replacement Table: Comprehensive

| Bad form | Replace with | Category |
|---|---|---|
| `заапплаить трансформацию` | `применить трансформацию` | verb anglicism |
| `матчить распределение` | `соответствовать распределению` | verb anglicism |
| `хэндлить ошибки` | `обрабатывать ошибки` | verb anglicism |
| `имплементировать / имплементить` | `реализовать / внедрить` | verb anglicism |
| `скейлить систему` | `масштабировать систему` | verb anglicism |
| `это импактит метрики` | `это влияет на метрики` | verb anglicism |
| `продакшен-реди` | `готов к деплою / готов к production` | hybrid adjective |
| `сетапить пайплайн` | `настроить / собрать пайплайн` | verb anglicism |
| `перформанс модели` | `качество модели / метрики модели` | bad transliteration |
| `фиксить проблему` | `исправлять проблему` | verb anglicism |
| `лейбл` | `метка` | bad transliteration |
| `фича` | `признак` | bad transliteration |
| `боттлнек` | `узкое место` | bad transliteration |
| `капасити модели` | `ёмкость модели` | bad transliteration |
| `трейд-офф` | `компромисс / баланс` | bad transliteration |
| `робастность` | `устойчивость` | bad transliteration |
| `тюнить параметры` | `настраивать / подбирать параметры` | verb anglicism |
| `оверсэмплить` | `увеличить долю примеров` | verb anglicism |
| `аугментить данные` | `применить аугментацию` | verb anglicism |

## 5) False Friends and Semantic Traps

These words look like they should translate one way, but the correct Russian is different.

| English source | Wrong Russian | Correct Russian | Why |
|---|---|---|---|
| production | производство | production / эксплуатация / деплой | `производство` = manufacturing |
| actual | актуальный | реальный / фактический | `актуальный` = current/relevant, not "actual" |
| policy | политика (bare) | политика аугментаций / аугментационный пайплайн | bare `политика` = government policy to Russian ears |
| transform | преобразование (everywhere) | трансформация in Albumentations prose | `преобразование` is fine in math, but `трансформация` is the Albumentations term |
| capacity | вместимость | ёмкость модели | `вместимость` = physical capacity of a container |
| aggressive | агрессивный (mechanically everywhere) | `сильный`, `жёсткий`, `тяжёлый` | `агрессивный` is often fine, but not always the best choice |
| collapse | коллапсировать | разваливаться / ломаться / резко проседать | `коллапсировать` is too medical/astrophysical |
| deploy | развернуть (everywhere) | `при деплое`, `в эксплуатации`, `в production` | `развернуть` is correct but overused |
| sensitive | чувствительный | `чувствителен к` in CV context | works, but sometimes `зависит от` is cleaner |
| domain | домен | предметная область / домен | `домен` is fine in ML context, `предметная область` in formal text |
| pipeline (data) | конвейер | пайплайн | `конвейер` means factory conveyor belt |
| performance | перформанс | качество / метрики / производительность | context-dependent |
| evidence | эвиденция | доказательства / результаты | `эвиденция` is not Russian |
| bias | биас | смещение | `биас` is acceptable in some ML contexts but `смещение` is cleaner |

## 6) English First Mention vs Lazy Carry-Over

Good:

- `grayscale (оттенки серого)` -> later `оттенки серого`
- `pixel-level (пиксельные) трансформации` -> later `пиксельные трансформации`
- `augmentation policy (политика аугментаций)` -> later one chosen stable form

Bad:

- `grayscale` repeated in every third paragraph
- `policy` left raw even though the article already established `политика аугментаций`
- alternating `baseline`, `базовая точка отсчёта`, and `baseline без аугментации`

The question is not "can the reader understand this English word?" The question is "does this English word still earn its place in a Russian sentence?"

## 7) Verb Policy

Prefer:

- Russian verb + technical noun
- direct verb over abstract noun
- concrete action over English-stem slang

Good:

- `собрать пайплайн`
- `ограничить интенсивность`
- `проверить абляциями`
- `прогнать валидацию`
- `подобрать силу аугментации`
- `вынуждать модель выучивать`
- `заполнить пробелы`
- `сократить разрыв`
- `расширить распределение`
- `уплотнить покрытие многообразия`

Bad:

- `зааблейтить`
- `зарегуляризить`
- `заинференсить`
- `матчить домен`
- `пофиксить`
- `задеплоить`
- `протюнить`

### Verb strength hierarchy

When you have a choice between a weak verb + abstract noun vs a strong verb, always pick the strong verb:

| Weak | Strong |
|---|---|
| `осуществить процесс обучения` | `обучить` |
| `произвести оценку` | `оценить` |
| `сделать проверку` | `проверить` |
| `выполнить применение трансформации` | `применить трансформацию` |
| `производить генерацию вариантов` | `генерировать варианты` |

## 8) Noun Policy

Do not keep an English noun in prose just because the source uses it.

Use Russian when it reads cleaner and stays precise:

- `метка`, not `лейбл`
- `признак`, not `фича`
- `узкое место`, not `боттлнек`
- `устойчивость`, not `робастность`
- `пропускная способность`, not `throughput` bare
- `размытие`, not `блюр`
- `шум`, not `нойз`
- `переобучение`, not `оверфиттинг`
- `ёмкость`, not `капасити`
- `разметчик`, not `аннотатор` (both acceptable, but `разметчик` is more natural)

Keep English only when it is truly entrenched or canonical:

- `pipeline` -> `пайплайн`
- `dataset` -> `датасет`
- `inference` -> `инференс`
- `CoarseDropout` -> keep exact
- `batch` -> `батч`
- `backbone` -> `бэкбон`

## 9) Sentence-Level Anti-Calque Rules

Common English-to-Russian failure pattern: the sentence is technically correct, but sounds like translated management English.

Prefer:

- stronger verbs,
- fewer abstract nouns,
- fewer stacked subordinate clauses,
- less passive voice,
- explicit cause-and-effect,
- main claim near the beginning.

### Rewrite patterns

| Weak calque | Better Russian |
|---|---|
| `это не редкий edge case` | `это не редкость` / `это не экзотика` |
| `модель эксплуатирует особенности данных` | `модель опирается на случайные особенности данных` |
| `аугментация является инструментом` | `аугментация помогает` / `аугментация работает как инструмент` |
| `это может иметь эффект на метрики` | `это влияет на метрики` |
| `данная секция рассматривает` | `в этом разделе разберём` |
| `модель коллапсирует на...` | `модель ломается на...` / `резко проседает на...` |
| `это случается потому, что модель...` | start directly: `Модель...` |
| `существует возможность того, что...` | `может случиться так, что...` / just state it directly |
| `представляется целесообразным...` | drop it |

### Passive-to-active conversions

English technical writing overuses passive voice. Russian is more direct.

| English passive | Bad Russian passive | Better Russian active |
|---|---|---|
| `The transform is applied...` | `Трансформация применяется...` | `Пайплайн применяет трансформацию...` |
| `Performance was degraded by...` | `Производительность была ухудшена...` | `X ухудшил метрики...` |
| `The label can be corrupted` | `Метка может быть испорчена` | `Это портит метку` / `Метка портится` |
| `It was observed that...` | `Было обнаружено, что...` | just state the fact |
| `The model is trained with...` | `Модель обучается с помощью...` | `Модель обучают с...` |

## 10) Paragraph-Level Anti-Calque Rules

### English scaffolding that Russian doesn't need

English technical writing often starts paragraphs with throat-clearing. Russian doesn't need it.

| English scaffold | Drop or replace |
|---|---|
| `It is important to note that...` | just state the fact |
| `It should be mentioned that...` | just state the fact |
| `In this section, we will discuss...` | `Здесь разберём...` or just start |
| `As previously mentioned...` | drop if it's obvious |
| `One common approach is to...` | `Один из подходов —` or just describe it |
| `There are several reasons why...` | just list the reasons |

### Paragraph shape problems from English

English docs paragraphs often delay the point. Russian Habr paragraphs should lead with the point.

English paragraph shape (mediocre):
1. Context sentence
2. More context
3. The actual claim
4. Consequence

Russian paragraph shape (better):
1. The claim or why the reader should care
2. Mechanism
3. Consequence or practical implication

## 11) Final Anti-Anglicism Sweep

Do one pass dedicated only to English leftovers.

Search for:

- raw Latin tokens in prose,
- repeated bilingual first mentions,
- English nouns outside code and links,
- mixed terminology for the same concept,
- headings that still sound like translated docs slots.

Ask:

- Is this English here because it is canonical?
- Is it entrenched in Russian ML usage?
- If I replace it with Russian, do I lose precision?
- If not, why is it still here?

## 12) Do Not Overcorrect

Avoid fake purity.

Do **not** force awkward Russian where ML usage is already stable:

- `конвейер` instead of `пайплайн`
- `увеличение данных` instead of `аугментация`
- `вывод` instead of `инференс`
- `точка сохранения` instead of `чекпоинт`
- `набор данных` everywhere instead of `датасет`
- `обратная связь` instead of `фидбэк` (debatable — `обратная связь` is fine when it's not about neural network backprop)

The goal is not maximum Russification. The goal is **native Russian technical prose** — the way a strong Russian ML engineer actually writes and speaks.

## 13) Manual Review Questions

Before finalizing a section, ask:

- Which English words stayed only because I followed the source too literally?
- Which hybrid verbs can be replaced with a Russian verb immediately?
- Would this paragraph sound normal if read aloud by a strong Russian ML engineer?
- Does the sentence contain too many English nouns in a row?
- Can I replace this English noun with Russian without losing precision?
- Am I keeping English because it is canonical, or because I was too lazy to rewrite?
- Did I add `данный`, `осуществлять`, `в рамках` anywhere? Remove them.

## 14) Automated Check

Use the utility linter for first-pass detection:

`python .cursor/skills/habr-russian-translation/scripts/term-lint.py <file-or-dir>`

Notes:

- It skips fenced code blocks and inline code spans.
- It flags explicit banned forms and mixed-script tokens.
- It is intentionally conservative: passing lint does **not** mean the prose is clean.
- Manual review is still required for raw English noun clusters and sentence-level calques.
- Run it early and often during translation — catching anglicisms in draft is cheaper than fixing them in review.
