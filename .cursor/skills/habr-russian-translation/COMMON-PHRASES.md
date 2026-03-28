# Common Phrase Translations for Habr

Use this file for **multi-word expressions, collocations, and recurring technical phrases**.

Single-term choices live in `GLOSSARY.md`.

The principle: **translate the meaning of the phrase, not the words in it.** English collocations rarely map word-for-word to natural Russian. When they do, the result usually sounds like a calque.

## 1) Core Phrases: Article-Narrative Style

| English phrase | Preferred Russian | Notes |
|---|---|---|
| default outcome | типичный результат | Not `дефолтный исход` |
| edge case | редкий крайний случай / редкость | Keep English only in informal aside |
| bridge the gap | сократить разрыв / закрыть разрыв | |
| narrow distribution | узкое распределение | |
| representative training data | репрезентативные обучающие данные | |
| preserve semantic meaning | сохранять семантику / смысл | Not `сохранять семантическое значение` |
| underlying visual task | сама зрительная задача / собственно задача распознавания | Depends on context |
| on the fly | на лету | Stable ML phrase |
| starter policy | стартовая политика аугментаций / стартовый пайплайн | |
| kitchen-sink policy | подход «суём всё подряд» / подход «добавили всё подряд» | Not literal; this is an idiom |
| hidden regression | скрытая регрессия | |
| silently damage performance | незаметно портить качество / метрики | |
| practical playbook | практический план действий | |
| standalone conclusion | самостоятельное заключение | Habr adaptation |
| without exception | без исключений | |
| is the foundation of | на этом строится идея / в основе лежит | Not `является основой` |
| operates at two levels | действует на двух уровнях | |
| a different design constraint | другое ограничение при проектировании | |
| the most common reason | чаще всего / самая частая причина | |
| from first principles | от основ / с нуля | Not `от первых принципов` |
| for a structured step-by-step process | для пошагового подхода | |
| this sounds obvious, but | звучит очевидно, но | |
| X claims that... | X означает, что... / X предполагает, что... | Avoid calque `X утверждает, что...` for transforms or concepts |
| too unfocused | слишком наугад / слишком бессистемно | Avoid dead-sounding `безадресно` in this context |

## 2) Augmentation-Specific Phrases

| English phrase | Preferred Russian |
|---|---|
| label preservation | сохранение метки |
| label corruption | порча меток / нарушение корректности меток |
| label noise | шум меток |
| target synchronization | синхронизация таргетов |
| in-distribution augmentation | in-distribution аугментация / аугментация внутри распределения |
| out-of-distribution augmentation | out-of-distribution аугментация / аугментация вне распределения |
| realistic variation | реалистичная вариация |
| unrealistic but label-preserving | нереалистичная, но сохраняющая метку |
| deployment variability | вариативность условий деплоя |
| data-generating process | процесс сбора данных / процесс порождения данных |
| failure mode | когда вредит / режим отказа |
| task-critical color information | критичная для задачи цветовая информация |
| target type | тип таргета |
| pixel-level transforms | пиксельные трансформации |
| spatial transforms | пространственные трансформации |
| environment simulation | симуляция среды / погодных условий |
| dropout-style augmentation | dropout-аугментация / аугментация с маскированием |
| horizontal flip | горизонтальное отражение | In prose prefer this over `горизонтальный флип`; keep `HorizontalFlip` for API name |
| vertical flip | вертикальное отражение | Same rule as above |
| policy tuning | настройка политики аугментаций |
| one-axis ablation | абляция по одной оси |
| no-augmentation baseline | baseline без аугментации |
| transform family | семейство трансформаций |
| augmentation strength | сила аугментации / интенсивность аугментации |
| augmentation pipeline | аугментационный пайплайн / пайплайн аугментаций |
| stochastic augmentation | стохастическая аугментация |
| deterministic preprocessing | детерминированный препроцессинг |
| augmentation leakage | просачивание аугментации (в оценку) |
| over-augmentation | переаугментация |
| conservative policy | консервативная политика |
| aggressive policy | агрессивная политика / сильная аугментация |
| always-on transform | постоянно включённая трансформация |
| label-preserving transform | трансформация, сохраняющая метку |
| label-breaking transform | трансформация, нарушающая метку |
| destroys the label | делает метку неверной / ломает метку |
| class-specific augmentation | классоспецифичная аугментация / аугментация по классам |
| per-domain policy | аугментация по доменам / policy по домену |
| targeted augmentation | прицельная аугментация |
| multi-target call | вызов с несколькими аннотациями |

## 3) Training and Evaluation Phrases

| English phrase | Preferred Russian |
|---|---|
| train from scratch | обучать с нуля |
| fine-tune a pretrained model | дообучать предобученную модель |
| overfit badly | сильно переобучаться |
| under control | под контролем |
| regularization pressure | регуляризационное давление |
| regularization budget | регуляризационный бюджет |
| validation metrics fluctuate | метрики валидации скачут |
| training loss plateaus high | training loss выходит на высокое плато |
| read metrics honestly | смотреть на метрики без самообмана / честно интерпретировать метрики |
| per-class metrics | метрики по классам |
| subgroup metrics | метрики по подгруппам |
| per-size-bin metrics | метрики по бинам размеров |
| synthetic stress-testing | синтетическое стресс-тестирование |
| failure slice | трудный срез / проблемное подмножество |
| hard example mining | отбор трудных примеров |
| model shortcut | shortcut модели / путь наименьшего сопротивления |
| calibration worsens | калибровка ухудшается |
| top-line metric | верхнеуровневая метрика / агрегированная метрика |
| aggregate metric | агрегированная метрика |
| outcome variance | дисперсия результатов |
| two seeds minimum | минимум два сида |
| controlled ablation | контролируемая абляция |
| confounded experiment | эксперимент со смешанными факторами |
| confidence miscalibration | раскалибровка уверенности |
| run at least two seeds | запускайте минимум два сида |
| lock policy before architecture sweeps | зафиксируйте augmentation policy перед перебором архитектур |

## 4) Optimization and Theory Phrases

| English phrase | Preferred Russian |
|---|---|
| decision boundary | граница решений |
| input-space stochasticity | стохастичность во входном пространстве |
| domain-shaped noise | шум, согласованный с предметной областью |
| semantically structured regularizer | семантически осмысленный регуляризатор |
| nuisance factor | фактор помех / несущественный фактор |
| smooth decision boundaries | сглаживать границы решений |
| low-dimensional manifold | низкоразмерное многообразие |
| densify the manifold | уплотнять многообразие / уплотнять покрытие |
| go off the manifold | уходить с многообразия |
| on or near the manifold | на многообразии или рядом с ним |
| symmetry group | группа симметрий |
| architecture prior | архитектурный априор |
| mathematically clean symmetry | чистая математическая симметрия |
| real-world variation | вариация реального мира |
| non-algebraic invariance | неалгебраическая инвариантность |
| invariance to nuisance factors | инвариантность к факторам помех |
| equivariance for spatial targets | эквивариантность для пространственных таргетов |
| coupled knobs | связанные ручки (настройки) |
| the model spends capacity | модель тратит ёмкость / ресурсы |
| reduce memorization pressure | ослаблять запоминание |
| interaction between transforms | взаимодействие трансформаций |
| transforms interact nonlinearly | трансформации взаимодействуют нелинейно |
| label-preservation boundary | граница сохранения метки |

## 5) Production and Operations Phrases

| English phrase | Preferred Russian |
|---|---|
| in production | в production / при деплое / в эксплуатации |
| deployment conditions | условия деплоя / эксплуатации |
| serving pipeline | инференс-пайплайн / пайплайн на сервисе |
| throughput bottleneck | узкое место по пропускной способности |
| accidental augmentation leakage | случайное просачивание аугментации в оценку |
| governed configuration | управляемая конфигурация |
| policy governance | управление политикой аугментаций |
| artifact metadata | метаданные артефакта |
| revisit the policy | пересмотреть политику |
| production-adjacent bug | ошибка, которая всплывает перед деплоем |
| version the policy | версионировать (политику) |
| mystery regression | загадочная регрессия |
| prefetch buffer | буфер предзагрузки |
| GPU utilization | утилизация GPU / загрузка GPU |
| data loader workers | воркеры загрузчика данных |
| epoch time increases | время эпохи растёт |
| cache deterministic preprocessing | кэшируйте детерминированный препроцессинг |
| ship a degraded model | выкатить модель с ухудшенным качеством |
| model drift | дрифт модели |
| annotation guidelines shift | правила разметки меняются |
| product constraints shift | продуктовые ограничения меняются |

## 6) Self-Supervised and Advanced Phrases

| English phrase | Preferred Russian |
|---|---|
| contrastive methods | контрастивные методы |
| contrastive pretraining | контрастивное предобучение |
| create multiple augmented views | создать несколько аугментированных вариантов |
| pull together representations | притягивать представления |
| push apart representations | отталкивать представления |
| baked into the representation | зашиты в представление |
| downstream task performance | качество на последующих задачах |
| domain randomization | Domain randomization |
| sim-to-real transfer | перенос из симуляции в реальность |
| wide enough training distribution | достаточно широкое обучающее распределение |
| test-time augmentation | Test-time augmentation (TTA) |
| aggregate predictions | объединить предсказания |
| ensemble of augmented views | ансамбль аугментированных вариантов |
| multi-scale inference | инференс в нескольких масштабах |
| inference cost | вычислительная стоимость инференса |

## 7) Habr-Friendly Rewrites for English Idioms

English idioms should be rewritten to carry the same meaning in natural Russian, not calqued word-for-word.

| English phrase | Do not calque | Preferred Russian |
|---|---|---|
| fire-and-forget | `огонь-и-забыл` | `настроил и забыл` |
| train hard, test easy | literal structure everywhere | `тренируйся на сложном, тестируйся на лёгком` |
| fill the gaps | `заполнить гэпы` | `заполнить пробелы` |
| go deeper | `пойти глубже` | `разобрать глубже` / `углубиться` |
| works out of the box | `работает из коробки` is okay, but do not overuse | `работает из коробки` / `запускается без допнастройки` |
| this is where X fails | `это где X ломается` | `именно здесь X ломается` |
| keep what helps | `оставьте то, что помогает` | `оставляйте только то, что реально помогает` |
| cheap to integrate | `дешёвый для интеграции` | `легко встроить` / `дёшево внедрить` |
| squeeze additional accuracy | `сжать дополнительную точность` | `выжать дополнительную точность` |
| from scratch | `от скрэтча` | `с нуля` |
| high-leverage tool | `высокорычажный инструмент` | `один из самых эффективных инструментов` |
| birdwatcher | `бёрдвотчер` | `орнитолог` / `наблюдатель за птицами` |
| hairline crack | `волосная трещина` | `узкая трещина толщиной с волос` |
| the training set captures a specific slice of reality | `обучающий набор захватывает специфический слайс реальности` | `датасет фиксирует лишь узкий срез реальности` |
| the model exploits those specifics | `модель эксплуатирует эти специфики` | `модель опирается на случайные особенности` |
| for a deeper treatment | `для более глубокого обращения` | `подробнее об этом — в разделе...` |
| in practice, you build... | `на практике вы строите...` | `на практике делают пайплайн, ...` (impersonal) |
| this is far more productive than | `это гораздо более продуктивно, чем` | `это гораздо продуктивнее, чем` |
| produces valid outputs that silently corrupt training | `производит валидные выходы, которые тихо портят обучение` | `даёт валидные выходы, которые тихо портят обучение` |

## 8) Phrases Better Rewritten Than Preserved

These are common cases where literal translation or lazy English carry-over makes the prose worse.

| Source pattern | Bad Russian | Better Russian |
|---|---|---|
| The model collapses on... | модель коллапсирует на... | модель ломается на... / резко проседает на... |
| The data captures a slice of reality | данные захватывают слайс реальности | датасет фиксирует лишь срез реальности |
| The model exploits specifics | модель эксплуатирует специфику | модель опирается на случайные особенности съёмки |
| the model spends capacity | `модель тратит капасити` | `модель тратит ёмкость / ресурсы` |
| without becoming a bottleneck | не становясь боттлнеком | не становясь узким местом |
| this advice is incomplete | этот совет инкомплит | этот совет неполон |
| fails catastrophically | катастрофически ломается | сильно проседает / перестаёт работать |
| the bug never raises an exception | баг никогда не вызывает эксепшен | баг при этом не вызывает никаких исключений |
| when chosen correctly | когда выбрано корректно | при правильном выборе |
| obviously yes | `очевидно, да` | `Очевидно да.` (no comma — this is a reaction, not a clause) |
| without hesitation | без колебаний | без сомнений |
| this section covers | данная секция покрывает | в этом разделе — |
| regardless of dataset size | вне зависимости от размера датасета | вне зависимости от размера датасета (this one works) |

## 9) Section-Label Rewrites

Section labels in English docs often use abstract nouns. For Habr, prefer claim-style or action-style labels.

| English section label | Bad literal | Better Habr heading |
|---|---|---|
| Failure mode: | Режим отказа: | Когда вредит: |
| What can go wrong: | Что может пойти не так: | Где это ломается: |
| Key takeaway: | Ключевой вывод: | Главный вывод: |
| Practical defaults | Практические значения по умолчанию | Практические значения по умолчанию (works) |
| Why each transform is there | Почему каждая трансформация включена | Что делает каждая трансформация |
| Task-Specific and Targeted Augmentation | Специфичная для задачи и целенаправленная аугментация | Аугментация под задачу |
| Expand the Policy Deliberately | Расширяйте политику обдуманно | Расширяйте политику обдуманно: семейства трансформаций |
| Know the Failure Modes Before They Hit Production | Знайте режимы отказа до production | Типичные ошибки: лучше знать до production |
| Evaluate With a Repeatable Protocol | Оценивайте с воспроизводимым протоколом | Оценка по воспроизводимому протоколу |
| Beyond Standard Training | За пределами стандартного обучения | За пределами стандартного обучения (works) |
| Production Reality: Operational Concerns | Production реальность: операционные вопросы | Production: эксплуатационные аспекты |
| Prevent Silent Label Corruption | Предотвратите тихую порчу меток | Синхронизация таргетов: как не испортить метки незаметно |

## 10) How to Use This File

- If the source phrase is technically recurrent, prefer a stable Russian collocation.
- If the English phrase is catchy but unnatural in Russian, rewrite the effect, not the words.
- If the phrase names a formal method or object, keep the canonical English name and explain it once in Russian.
- When multiple Russian options are listed, pick one per article and keep it consistent.
- Do not mix English and Russian versions of the same phrase within one article.
