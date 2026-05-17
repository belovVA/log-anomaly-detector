# Обнаружение аномалий в логах веб-сервера

 Проект по **unsupervised-детекции аномалий** в логах распределённой системы. В качестве данных используется публичный датасет **HDFS** (логи Hadoop Distributed File System) — типичный пример инфраструктурных логов, близкий по структуре к логам веб- и backend-сервисов (временная метка, уровень, компонент, сообщение).

## Содержание

- [Описание работы](#описание-работы)
- [Принцип работы](#принцип-работы)
- [Структура репозитория](#структура-репозитория)
- [Требования](#требования)
- [Установка](#установка)
- [Запуск](#запуск)
- [ClearML и отчёт](#clearml-и-отчёт)
- [Результаты](#результаты)
- [Ограничения](#ограничения)

## Описание работы

Цель — построить пайплайн, который **без размеченных аномалий** на этапе обучения:

1. Загружает и нормализует строки логов.
2. Обучает **нейросетевой encoder** (BiLSTM) представлять логи в виде компактного вектора.
3. Строит **Isolation Forest** на эмбеддингах и выделяет строки с необычно высоким anomaly score.
4. Проверяет качество на **синтетическом наборе** с известными метками (норма / аномалия).
5. Логирует эксперимент и визуализации в **ClearML**, формирует **PDF-отчёт** с Embed-графиками.

Основной артефакт — Jupyter-ноутбук [`./main.ipynb`](./main.ipynb).

## Принцип работы

```text
Сырая строка лога
       │
       ▼
Нормализация (IP → [IP], block id → [B], длинные числа → [NUM], …)
       │
       ▼
Токенизация + словарь (vocabulary)
       │
       ▼
BiLSTM Encoder  ──►  вектор z (эмбеддинг строки)
       │                    │
       │ (self-supervised   │
       │  seq2seq на train) ▼
       │              Isolation Forest (train только на «норме»)
       │                    │
       │                    ▼
       │              anomaly score + порог (95-й перцентиль train)
       ▼
score > порог  →  аномалия
```

### Этапы подробнее

| Этап | Метод | Зачем |
|------|--------|--------|
| Предобработка | Regex-нормализация, lower-case, фильтрация шумовых токенов | Убрать «шум» (IP, id блоков), сохранить структуру сообщения |
| Encoder | Embedding + BiLSTM + pooling + Dense; обучение seq2seq (восстановление токенов) | Получить семантическое представление строки лога без ручной разметки |
| Детектор | Isolation Forest на векторах `z` | Классическая unsupervised-модель для выбросов в пространстве признаков |
| Порог | 95-й перцентиль score на train | Контроль доли «подозрительных» строк (~5%) |
| Валидация | Синтетика: 300 нормальных + 100 аномалий (6 типов) | Объективные precision / recall / F1 при известном ground truth |

### Типы синтетических аномалий

- `foreign_format` — формат не HDFS (например, access-log веб-сервера)
- `warn_panic` — WARN и panic/timeout
- `unknown_component` — неизвестный компонент / ERROR
- `garbage` — бессмысленная строка
- `too_short` — слишком мало токенов
- `mutated_normal` — почти нормальный HDFS-лог с вставками FATAL/exception

## Структура репозитория

```text
.
├── README.md # описание проекта
├── dataset
│   ├── HDFS_2k.log # датасет
│   ├── hdfs_encoder.keras  # сохранённый encoder (после обучения)
│   ├── hdfs_preprocessed_data.pkl # снимок результатов первого этапа
│   └── run_artifacts.pkl # порог, словарь (после прогона)
├── docs
│   ├── generate_clearml_report_embeds.py # генерация Markdown для ClearML Report
│   ├── report.pdf  # финальный PDF
│   └── report_embeds.md  # тело отчёта с iframe Embed
└── main.ipynb # основной пайплайн
```

## Требования

- **Python** 3.9+
- **Jupyter** (Notebook или VS Code / Cursor с kernel `.venv`)
- Аккаунт **[ClearML](https://app.clear.ml)** (бесплатный cloud) — для логирования и отчёта
- ~2 GB места на диске (TensorFlow + зависимости)

### Python-пакеты

```text
numpy pandas matplotlib seaborn scikit-learn tensorflow clearml
```

## Установка

```bash
cd /path/to/your-project

# виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow clearml
```

### Настройка ClearML (один раз)

```bash
clearml-init
```

Вставьте credentials с [app.clear.ml](https://app.clear.ml) → **Settings → Workspace → Create credentials**.

Конфиг сохранится в `~/clearml.conf`.

## Запуск

### 1. Jupyter / Cursor

1. Откройте [`example/ML.ipynb`](example/ML.ipynb).
2. Выберите kernel: **`.venv`** .
3. Выполните ячейки **сверху вниз** (Run All).

Порядок в ноутбуке:

| Блок | Содержание |
|------|------------|
| 1–6 | Загрузка HDFS, EDA, нормализация, словарь |
| ClearML init | `Task.init`, проект `/log-anomaly-detector` |
| 7–12 | Encoder, Isolation Forest, демо-детекция |
| 13–14 | Графики score и PCA (test) |
| 15–16 | Синтетический датасет и оценка (F1, confusion matrix) |
| 17 | Загрузка артефактов в ClearML |

Первый прогон encoder на CPU занимает **несколько минут**.

### 2. Google Colab (опционально)

1. Загрузите `ML.ipynb` в Colab.
2. В начале ноутбука: `!pip install clearml tensorflow scikit-learn …`
3. `!clearml-init` или Secrets: `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY`.
4. Run all.

### 3. Детекция одной новой строки

После прогона ноутбука доступна функция:

```python
result = detect_anomaly_line("081111 235959 9999 WARN dfs.FSNamesystem: ...")
print(result['is_anomaly'], result['score'], result['threshold'])
```

## ClearML и отчёт

### Эксперимент

- **Project:** `log-anomaly-detector`
- **Scalars:** `encoder/loss`, `synth_f1`, `synth_recall`, …
- **Plots:** `hdfs_eval/score_distribution`, `hdfs_eval/pca_embeddings`, `synthetic_eval/dashboard`

### Финальный PDF-отчёт

Требования к сдаче: отчёт только через **ClearML Reports**, графики — **Embed** (`<iframe>`), файл **`docs/report.pdf`**.

```bash
source .venv/bin/activate
python example/generate_clearml_report_embeds.py
```

1. Скопируйте содержимое [`docs/report_embeds.md`](docs/report_embeds.md).
2. [app.clear.ml](https://app.clear.ml) → **Reports** → **+ NEW REPORT** → вставить → **Publish**.
3. **Download PDF** → сохранить как **`docs/report.pdf`**.

Подробно: [`docs/CLEARML_REPORT_STEPS.md`](docs/CLEARML_REPORT_STEPS.md).

## Результаты

На синтетическом тесте (400 строк, ground truth известен), типичные метрики последнего прогона:

| Метрика | Значение |
|--------|----------|
| Accuracy | ~0.95 |
| Precision | ~0.95 |
| Recall | ~0.84 |
| F1 | ~0.89 |
| ROC-AUC | ~0.93 |

**Выводы:**

- Модель **хорошо** отделяет «очевидные» аномалии (чужой формат, мусор, WARN/panic).
- Слабее на **`mutated_normal`** — строках, похожих на обычный HDFS.
- Подход пригоден для **первичного скрининга** подозрительных логов; для продакшена нужна валидация на реальных инцидентах.

Полная аналитика — в [`docs/report.pdf`](docs/report.pdf).

## Ограничения

- Обучение **без разметки** только на «нормальных» HDFS-логах; порог зависит от train.
- Датасет **HDFS_2k** (2000 строк) — учебный масштаб, не полный LogHub.
- Isolation Forest предполагает, что аномалии — **редкие выбросы** в пространстве эмбеддингов.

## Датасет

[HDFS_2k.log](https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log) — подмножество [LogHub / HDFS](https://github.com/logpai/loghub/tree/master/HDFS).

## Авторы
Белов Владимир Алексеевич \
Хуснутдинов Азамат Эдуардович \
Учебный проект по дисциплине "Прикладные методы искусственного интеллекта" \
КНИТУ-КАИ им. А.Н. Туполева 

Казань 2026
