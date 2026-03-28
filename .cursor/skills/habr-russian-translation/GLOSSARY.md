# Glossary and Preferred Term Choices

Use this glossary as the default lexical policy for **English -> Russian Habr** localization.

The main question is not "can this English word stay?" but:

1. Is it canonical code/API vocabulary?
2. Is it an established Russian ML term?
3. Does clean Russian lose precision here?
4. If not, rewrite into normal Russian.

## 1) Term Decision Ladder

For every term, choose one of three buckets:

1. **Keep in English exactly** if it is a transform/class/API/method/metric/code token.
2. **Keep as established Russian ML transliteration** if that is how strong Russian practitioners normally write it.
3. **Translate into Russian** if a precise and natural Russian equivalent exists.

If you hesitate, prefer **better Russian prose** over lazy English carry-over.

## 2) Established ML Terms You Usually Keep

These are normal in Russian ML/CV writing and do not need forced de-anglicization. Trying to replace these with "pure Russian" sounds artificial and makes the text harder to read for the target audience.

| English | Russian default | Notes |
|---|---|---|
| augmentation | аугментация | Not `увеличение данных` |
| pipeline | пайплайн | Not `конвейер` in ML prose |
| dataset | датасет | `набор данных` is more formal and slower |
| inference | инференс | Avoid ambiguous `вывод` |
| deploy / deployment | деплой / при деплое | `развёртывание` is acceptable when tone is more neutral |
| crop | кроп | Verb can be `кадрировать`, but `кроп` is fine in CV prose |
| resize | ресайз | Verb can be `изменить размер` |
| backbone | бэкбон | Common in CV |
| checkpoint | чекпоинт | |
| config | конфиг | |
| worker | воркер | Data-loader context |
| ablation | абляция | Established |
| validation | валидация | Established |
| regularizer | регуляризатор | Established |
| regularization | регуляризация | Established |
| invariance | инвариантность | Established |
| equivariance | эквивариантность | Established |
| fine-tuning | fine-tuning / дообучение | In headings, English is acceptable; in prose, `дообучение` is often cleaner |
| target | таргет | Acceptable in augmentation-target compatibility context |
| anchor | анкор | Detection context |
| seed | сид | `random seed` context |
| contrastive | контрастивный | `контрастивное обучение` |
| interpolation | интерполяция | Established |
| stochastic | стохастический | Established |

## 3) Translate into Proper Russian

These usually sound better in proper Russian than in rough transliteration. Using the English form in polished prose marks the text as lazy translation.

| English | Preferred Russian | Avoid |
|---|---|---|
| label | метка | `лейбл` |
| feature | признак | `фича` in article prose |
| bottleneck | узкое место | `боттлнек` |
| throughput | пропускная способность | `фрупут` |
| forward pass | прямой проход | `форвард пасс` |
| backward pass | обратный проход | `бэкворд пасс` |
| data loader | загрузчик данных | `дата лоадер` |
| overfitting | переобучение | `оверфиттинг` |
| underfitting | недообучение | `андерфиттинг` |
| blur | размытие | `блюр` in narrative prose |
| noise | шум | |
| model capacity | ёмкость модели | `капасити` |
| robustness | устойчивость | `робастность` unless strict statistical context |
| trade-off | компромисс / баланс | `трейд-офф` in final polished prose |
| hard example mining | отбор трудных примеров | naked English phrase |
| decision boundary | граница решений | literal English carry-over |
| deployment conditions | условия деплоя / эксплуатации | `production conditions` left raw |
| magnitude | интенсивность / сила | `магнитуда` in augmentation context |
| severity | сила / интенсивность | `северити` |
| annotation | аннотация / разметка | choose one per article |
| annotator | разметчик / аннотатор | `аннотатор` is acceptable but `разметчик` is more natural |
| camera roll | наклон камеры | `ролл камеры` |
| edge case | крайний случай / редкость | `эдж кейс` |
| failure mode | режим отказа / «когда вредит» | `фейлюр мод` |
| shortcut (model) | путь наименьшего сопротивления | `шорткат модели` is acceptable in mixed context |
| domain shift | сдвиг домена | `домен шифт` |
| domain gap | разрыв между доменами | `домен гэп` |
| data-generating process | процесс сбора данных / порождения данных | naked English |
| connected component | связная компонента | established math term |
| confidence | уверенность (модели) | `конфиденс` |
| calibration | калибровка | established |
| smart/clever | умный / хитрый | `смарт` |
| plug-and-play | готов к использованию | `плаг-энд-плей` |
| sample | пример / сэмпл | `сэмпл` is weak; `пример` or `объект` depending on context |
| mask | маска | established |
| bounding box | ограничивающий бокс / рамка | `баундинг бокс` is acceptable but verbose |
| keypoint | ключевая точка | `кейпоинт` is unnecessary |
| variance | дисперсия | established math term |
| batch | батч | established in training context |
| epoch | эпоха | established |

## 4) Keep in English Exactly

Never localize these in prose unless you are explicitly explaining them:

- Albumentations transform/class names: `CoarseDropout`, `HorizontalFlip`, `RandomResizedCrop`, `Affine`, `Perspective`, `OpticalDistortion`, `SquareSymmetry`, `ConstrainedCoarseDropout`, `ToGray`, `ChannelDropout`, `ColorJitter`, `PlanckianJitter`, `GaussianBlur`, `MedianBlur`, `MotionBlur`, `GaussNoise`, `RandomBrightnessContrast`, `PhotoMetricDistort`, `RandomErasing`, `GridDropout`, `RandomRain`, `RandomFog`, `RandomSunFlare`, `RandomShadow`
- Method names and model families: SimCLR, MoCo, BYOL, DINO, MixUp, CutMix, Mosaic, Copy-Paste
- Metric tokens: mAP, IoU, F1, ROC-AUC
- Abbreviations: GPU, CPU, TTA, OCR, JPEG, H&E, SSR
- Code tokens and parameter names: `p=0.5`, `image`, `bboxes`, `mask`, `keypoints`, `label_mapping`, `label_fields`, `keypoint_params`
- Literal config and API names: `weight decay`, `label smoothing`, `stochastic depth`
- Training regime terms when used technically: `learning rate`, `learning rate schedule`, `training loss`

## 5) Ambiguous Terms: Preferred Handling

These often need more thought than a simple keep/translate rule. The correct choice depends on context, tone, and what is nearby in the sentence.

| English | Preferred handling | Why |
|---|---|---|
| baseline | `базовый ориентир`, `базовая точка отсчёта`, optionally `baseline` on first mention | Naked English often looks lazy; but `baseline без аугментации` is cleaner than the full Russian |
| policy | `политика аугментаций`, `аугментационный пайплайн` | Bare `policy` is weak Russian; `augmentation policy` in mixed context is tolerable if you've already established the term |
| production | `production`, `при деплое`, `в эксплуатации` | Avoid `производство` at all costs — false friend |
| grayscale | `оттенки серого`, optionally `grayscale (оттенки серого)` on first mention | Do not repeat English every time; after first mention use only Russian |
| train/test distribution gap | `расхождение между train и test распределениями`, `разрыв между обучением и деплоем` | Depends on precision needed |
| self-supervised learning | `self-supervised learning`, optionally `самообучение без разметки` as explanation | Keep English if referencing the field name |
| contrastive learning | `contrastive learning`, optionally `контрастивное обучение` | Russian calque is acceptable, choose one and stay consistent |
| downstream task | `downstream-задача` or `последующая задача` | Choose based on tone and precision |
| learning rate | `learning rate`, optionally `скорость обучения` on first mention | Both acceptable; keep one variant per article |
| failure slice | `трудный срез`, `срез ошибок`, `сложное подмножество` | Pick by context, not literally |
| transform | `трансформация` in Albumentations prose; `преобразование` in broader theory | Both correct, choose by context and stick with it |
| dropout (as augmentation) | `dropout` | Not the same as network dropout; keep English to avoid confusion |
| dropout (as network regularizer) | `dropout` | Established English term |
| data augmentation | `аугментация данных` or `аугментация изображений` | Not `увеличение данных` |
| label noise | `шум меток` | Not `лейбл нойз` |
| class-specific | `классоспецифичный` or `по классам` | Both acceptable |
| per-domain | `по доменам` | Not `per-domain` bare |
| hard negative | `трудный негативный пример` | Not `хард негатив` |
| test-time augmentation | `Test-time augmentation (TTA)` on first mention, then `TTA` | Keep acronym English |
| domain randomization | `Domain randomization` on first mention, then mixed or Russian explanation | Keep as proper noun |
| self-driving | `автономное вождение` | Not `селф-драйвинг` |
| camera trap | `камера-ловушка` | Standard Russian term |
| overhead | `накладные расходы` / `дополнительная стоимость` | Not `оверхед` in polished prose |

## 6) Common English -> Russian Defaults in Technical Prose

These are not code tokens. In article prose they usually want Russian defaults, even if the source keeps them in English.

| English | Preferred Russian default | Notes |
|---|---|---|
| guide | статья / руководство / разбор | Depends on genre; Habr usually wants `статья` or `разбор` |
| section | раздел | Not `секция` in article prose |
| chapter | глава / большой раздел | `глава` is fine for long-form structure |
| hook | заход / зацепка / сильный заход | Choose by tone; often rewrite instead of translating literally |
| takeaway | главный вывод / практический вывод | Not `тейкауэй` |
| scope | рамки статьи / что именно разбираем | Rewrite by context |
| roadmap | план / маршрут по статье | Usually `План:` is enough |
| workflow | процесс / порядок работы / workflow | Bare English is often lazy outside tooling contexts |
| starter | стартовый / базовый | Avoid naked `starter` |
| conservative | консервативный / осторожный | Depends on tone |
| aggressive | агрессивный / сильный / жёсткий | Pick by context, not mechanically |
| practical | практический | Usually translates cleanly |
| production story | путь до production / история про деплой | Usually rewrite, not literal translate |
| standalone | самостоятельный / автономный | `самостоятельный` for article/conclusion |
| target audience | целевая аудитория / для кого текст | Usually rewrite by context |
| reader-facing | ориентированный на читателя | Usually rewrite by effect |

## 7) First-Mention Patterns

Use bilingual first mention only when it prevents ambiguity. After the first mention, drop the English or use the shorter form.

Pattern:

- `English term (русское пояснение)` on first mention
- shorter Russian or shorter mixed form later

Good examples:

- `grayscale (оттенки серого)` — then `оттенки серого` later
- `pixel-level (пиксельные) трансформации` — then `пиксельные трансформации` later
- `spatial (пространственные) трансформации` — then `пространственные трансформации` later
- `augmentation policy (политика аугментаций)` — then `augmentation policy` or `политика аугментаций` interchangeably
- `in-distribution (вариации внутри обучающего распределения)` — then `in-distribution` later
- `out-of-distribution (вариации вне обучающего распределения)` — then `out-of-distribution` later
- `CoarseDropout` — no explanation needed, it is code

Bad examples:

- repeating bilingual forms in every paragraph (reader fatigue),
- keeping English even after the concept is already clear (lazy),
- translating canonical transform names into Russian (wrong),
- providing bilingual for terms that need no explanation (`аугментация (augmentation)` — reader knows this).
- leaving the English in place forever after a helpful first mention.

## 8) Terms That Usually Need Russian After First Mention

These may appear in English in the source, but in polished Russian prose they usually should not stay naked for the whole article.

| English source form | Good first mention | After that |
|---|---|---|
| grayscale | `grayscale (оттенки серого)` | `оттенки серого` |
| pixel-level | `pixel-level (пиксельные)` | `пиксельные` |
| spatial | `spatial (пространственные)` | `пространственные` |
| starter policy | `стартовая политика аугментаций` | `политика` / `пайплайн` |
| failure mode | `режим отказа` or better rewritten heading | `когда вредит` / contextual Russian |
| baseline | `baseline без аугментации` or `базовая точка отсчёта` | one stable chosen form |
| workflow | `workflow (порядок работы)` only if needed | `процесс` / `порядок работы` |
| scope | contextual Russian | contextual Russian |
| takeaway | contextual Russian | contextual Russian |

If such words keep appearing in English paragraph after paragraph, you are following the source too literally.

## 9) Frequent Term Pairs from CV/ML Domain

| English | Preferred Russian |
|---|---|
| label preservation | сохранение метки |
| label corruption | порча меток / нарушение корректности меток |
| target synchronization | синхронизация таргетов |
| no-augmentation baseline | baseline без аугментации |
| one-axis ablation | абляция по одной оси |
| real-world failure slice | реальный трудный срез / проблемное подмножество |
| starter policy | стартовая политика аугментаций / базовый пайплайн |
| deployment variability | вариативность условий деплоя |
| semantic content | семантика / смысл изображения |
| physically grounded variation | физически правдоподобная вариация |
| hidden regression | скрытая регрессия |
| task signal | сигнал задачи |
| model shortcut | shortcut модели / путь наименьшего сопротивления |
| regularization budget | регуляризационный бюджет |
| regularization pressure | регуляризационное давление |
| training distribution | обучающее распределение |
| test distribution | тестовое распределение |
| deployment distribution | распределение при деплое |
| data manifold | многообразие данных |
| image manifold | многообразие изображений |
| pixel space | пиксельное пространство |
| semantic corruption | семантическая порча |
| silent corruption | незаметная порча / тихая порча |
| adversarial perturbation | состязательное возмущение |
| spatial target | пространственный таргет / пространственная аннотация |
| nearest-neighbor interpolation | nearest-neighbor интерполяция |
| boundary F1 | boundary F1 (keep) |
| mean IoU | средний IoU |
| per-class IoU | IoU по классам |
| label semantics | семантика меток |
| object of interest | объект интереса |
| geometric deep learning | geometric deep learning (keep as field name) |
| camera stack | камера / фотосистема |
| barrel distortion | бочкообразное искажение |
| pincushion distortion | подушкообразное искажение |

## 10) Stability Rule

For each article, choose one preferred variant per concept and keep it stable.

Examples:

- `ёмкость модели` everywhere, not `мощность модели` in one place and `капасити` in another.
- `политика аугментаций` everywhere, not `policy`, `augmentation policy`, and `аугментационный сетап` mixed together.
- `сохранение метки` everywhere, not alternating with `сохранение лейбла`.
- `переобучение` everywhere, not `оверфиттинг` in one place and `переобучение` in another.
- `трансформация` or `преобразование` — pick one per article, not mixed.
- `размытие` in prose, not `блюр` in one paragraph and `размытие` in the next.

When you find yourself switching between variants mid-article, go back and normalize.

## 11) Terms That Seem Translatable But Should Stay English

These terms tempt you to translate, but the Russian equivalent is either ambiguous, awkward, or not established:

| English | Why keep it | Wrong translation |
|---|---|---|
| pipeline | `конвейер` means something different in Russian engineering | `конвейер обработки` |
| dataset | `набор данных` is formal and verbose | `набор данных` is acceptable in formal writing only |
| inference | `вывод` means both "inference" and "conclusion" | `вывод модели` is ambiguous |
| augmentation | `увеличение данных` is a calque from data augmentation | `увеличение данных` |
| dropout (technique) | `выбрасывание` loses the connection to the canonical technique | `выбрасывание нейронов` |
| batch | `пакет` sounds like a postal package | `пакет обучающих данных` |
| epoch | could be `эпоха`, which is actually fine — but `эпоха` is also standard Russian | `эпоха` is correct |
| benchmark | `бенчмарк` is established | `тестовый стенд` |
| callback | `колбэк` is established in programming | `обратный вызов` is correct but less common in ML |

## 12) Translation Ledger Template

Before drafting, create a mini-ledger for the article:

| Concept | Preferred form | Allowed English? | First mention | Forbidden variants |
|---|---|---|---|---|
| policy | политика аугментаций | only if justified | plain Russian | `policy`, `сетап`, `augmentation policy` mixed randomly |
| label | метка | no | plain Russian | `лейбл` |
| robustness | устойчивость | only in strict statistical context | plain Russian | `робастность` in generic prose |
| grayscale | оттенки серого | first mention only | bilingual | repeated naked `grayscale` |

This takes two minutes and prevents half of the later cleanup work.

## 13) Domain-Specific Sub-Glossaries

### Medical imaging
| English | Russian |
|---|---|
| H&E staining | окрашивание H&E |
| histological section | гистологический срез |
| slide preparation | подготовка препарата |
| soft tissue | мягкие ткани |
| dermatoscope | дерматоскоп |
| skin lesion | кожное поражение |
| scanner hardware | оборудование сканера |

### Autonomous driving
| English | Russian |
|---|---|
| self-driving | автономное вождение |
| perception system | система восприятия |
| driving scene | сцена вождения |

### Remote sensing
| English | Russian |
|---|---|
| aerial imagery | аэросъёмка |
| satellite imagery | спутниковая съёмка |
| north-up convention | ориентация «север вверх» |

### Industrial inspection
| English | Russian |
|---|---|
| defect detection | обнаружение дефектов |
| micro-structure | микроструктура |
| industrial inspection | промышленная инспекция |
