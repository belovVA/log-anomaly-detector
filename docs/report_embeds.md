# Отчёт: обнаружение аномалий в логах HDFS
## 1. Постановка задачи
Flexible flow / логи HDFS: encoder (BiLSTM) + IsolationForest на эмбеддингах. Оценка на синтетическом датасете с известными метками.

## 2. Обучение encoder (loss)
### `encoder` / `loss`

<iframe src="https://app.clear.ml/widgets/?objectType=task&xaxis=iter&type=scalar&metrics=encoder&variants=loss&objects=79bc628903a3423a83d13013e2c4b9ab" width="100%" height="420" frameborder="0"></iframe>
### `encoder` / `val_loss`

<iframe src="https://app.clear.ml/widgets/?objectType=task&xaxis=iter&type=scalar&metrics=encoder&variants=val_loss&objects=79bc628903a3423a83d13013e2c4b9ab" width="100%" height="420" frameborder="0"></iframe>

## 3. Визуализации HDFS (test)

### `hdfs_eval` / `score_distribution`

<iframe src="https://app.clear.ml/widgets/?objectType=task&xaxis=iter&type=plot&metrics=hdfs_eval&variants=score_distribution&objects=79bc628903a3423a83d13013e2c4b9ab" width="100%" height="420" frameborder="0"></iframe>
### `hdfs_eval` / `pca_embeddings`

<iframe src="https://app.clear.ml/widgets/?objectType=task&xaxis=iter&type=plot&metrics=hdfs_eval&variants=pca_embeddings&objects=79bc628903a3423a83d13013e2c4b9ab" width="100%" height="420" frameborder="0"></iframe>

## 4. Синтетический датасет (ground truth)

### Метрики (из эксперимента ClearML)

- **threshold:** -0.0000
- **synth_precision:** 0.955
- **synth_recall:** 0.840
- **synth_f1:** 0.894
- **synth_accuracy:** 0.950
- **synth_roc_auc:** 0.933
- **synth_pr_auc:** 0.900
- **synth_fp:** 4.000
- **synth_fn:** 16.000
- **hdfs_test_anomaly_rate_pct:** 8.500

### `synthetic_eval` / `dashboard`

<iframe src="https://app.clear.ml/widgets/?objectType=task&xaxis=iter&type=plot&metrics=synthetic_eval&variants=dashboard&objects=79bc628903a3423a83d13013e2c4b9ab" width="100%" height="480" frameborder="0"></iframe>

### Сводка метрик (синтетика, 400 строк: 300 норм / 100 аномалий)

| Метрика | Значение | Интерпретация |
|--------|----------|---------------|
| **Accuracy** | 0.950 | Доля верных предсказаний на размеченной синтетике |
| **Precision** | 0.955 | Среди срабатываний «аномалия» — доля реальных аномалий |
| **Recall** | 0.840 | Доля найденных аномалий от всех размеченных |
| **F1** | 0.894 | Баланс precision и recall |
| **ROC-AUC** | 0.933 | Качество ранжирования по score (независимо от порога) |
| **PR-AUC** | 0.900 | То же с учётом дисбаланса классов |
| **Порог (train)** | -0.0000 | 95-й перцентиль anomaly score на HDFS train |

| | Pred Normal | Pred Anomaly |
|--|--|--|
| **True Normal** | 296 | 4 |
| **True Anomaly** | 16 | 84 |


### Интерпретация графика `synthetic_eval / dashboard`

**1. Гистограмма anomaly score (норма vs аномалия)**  
Нормальные строки (синий) в основном **левее порога** — низкий score. Аномалии (красный) смещены **вправо**, но хвосты **пересекаются**: часть аномалий похожа на типичный HDFS, часть нормальных получает повышенный score. Порог (~-0.0000) в целом разделяет классы, но не идеально.

**2. Confusion matrix**  
Около **99% нормальных** распознаны верно; **~84% аномалий** пойманы. Ошибок пропуска (FN) больше, чем ложных тревог (FP): модель **консервативнее** — реже помечает норму как аномалию, но **иногда пропускает** тонкие аномалии.

**3. Recall по типу аномалии (третья панель дашборда)**  
Для каждого типа: доля строк-аномалий этого типа, которые модель пометила как аномалии.

- Столбцы **≈ 1.0** — «явные» аномалии (`foreign_format`, `warn_panic`, `garbage`, `too_short`, `unknown_component`): score сильно выше порога.
- Столбец **ниже** — чаще всего **`mutated_normal`**: строка похожа на обычный HDFS, score низкий → пропуск (в прогоне: 16 из 100 FN — в основном этот тип).
- **Overall recall 0.84** — среднее по всем аномалиям; по типам значения могут отличаться.

### Интерпретация графиков HDFS (test)

**`hdfs_eval / score_distribution`** — распределение score на train/test; красная линия — порог. Большинство test-строк ниже порога (норма); хвост — кандидаты в аномалии (~5% по design `contamination=0.05`).

**`hdfs_eval / pca_embeddings`** — 2D-проекция эмбеддингов encoder. Аномалии (красные) частично выделяются в пространстве признаков, но есть **перекрытие** с нормой — это объясняет ошибки на «тонких» аномалиях.

## 5. Общая оценка качества модели

**Итог: модель показала себя хорошо для учебной unsupervised-постановки.**

1. **Encoder (BiLSTM) + IsolationForest** на эмбеддингах даёт **устойчивое** разделение нормы и аномалий на контролируемом синтетическом наборе (F1 ≈ 0.894, ROC-AUC ≈ 0.933).
2. Подход **работает** для «очевидных» аномалий (чужой формат, мусор, WARN/panic, слишком короткие строки).
3. **Слабые места:** обучение без разметки только на HDFS; порог с train; пропуски на **`mutated_normal`** и пограничные FP на нормальных логах.
4. Для промышленного использования нужна **валидация на реальных инцидентах** и настройка порога под цель (минимум FP vs минимум FN).

## 6. Выводы

- Пайплайн (нормализация → encoder → IsolationForest) **пригоден** для первичного скрининга подозрительных логов HDFS.
- Синтетическая оценка с известными метками подтверждает **высокое качество ранжирования** (ROC-AUC) и **практичный** баланс precision/recall.
- Рекомендация: развивать детекцию «тонких» аномалий (доп. признаки, разметка, другой порог или supervised-дообучение на эмбеддингах).
