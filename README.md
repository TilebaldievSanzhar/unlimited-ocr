# unlimited-ocr-bench

Лёгкий стенд для сравнения **Baidu Unlimited-OCR** против **Marker** на твоих документах
(в первую очередь — инвойсы с длинными таблицами line items, где marker ломает структуру).

Сравнивается **только сырой OCR → markdown**. Этап классификации/экстракции здесь намеренно не трогаем.

## Порты (общий сервер)

Сервис слушает порт из выделенного диапазона **7700–7799**:

| Сервис | Порт по умолчанию |
|---|---|
| Веб + REST API (этот стенд) | **7700** |
| SGLang (опционально, прод-подача) | **7701** |

Поменять порт стенда — переменной `APP_PORT` (оставайся в 7700–7799).

## Требования

- Linux + NVIDIA GPU + **NVIDIA Container Toolkit** (`nvidia-ctk`) для проброса GPU в Docker.
- Blackwell (RTX 5090, sm_120): образ собран на CUDA 12.9 + torch cu129 — иначе `no kernel image for sm_120`.
- Память: модель ~7.3 GB в bf16 + активации. На 5090 с ~20 GB свободными — с запасом, квантизация не нужна.

## Запуск через Docker (основной путь)

```bash
git clone https://github.com/TilebaldievSanzhar/unlimited-ocr.git
cd unlimited-ocr

docker compose up -d --build      # собрать и поднять
docker compose logs -f            # следить (первый запуск качает веса ~ГБ)
```

Открой `http://<server>:7700/`.

Другой порт в своём диапазоне:

```bash
APP_PORT=7705 docker compose up -d --build
```

Остановить / пересобрать:

```bash
docker compose down
docker compose up -d --build
```

**Что куда монтируется:**
- `./data` → выходы каждого прогона (`data/outputs/<timestamp>/` с `unlimited.md`, `marker.md`, страницами) видны на хосте.
- именованный том `hf-cache` → веса модели кэшируются и не качаются заново при рестарте.

> Если на твоей версии Docker GPU не подхватывается через `deploy.devices`, замени блок `deploy:`
> в `docker-compose.yml` на `runtime: nvidia` (или запусти `docker run --gpus all ...`).
> Проверь хост: `docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi`.

## Использование

### Веб (ручное сравнение)

Загрузи PDF инвойса → выбери режим (`base` — точный 1024px; `gundam` — быстрый 640px) →
при желании подгрузи `.md` из текущего marker. Получишь две колонки с raw markdown и метриками.

Ключевая метрика для инвойсов — **«макс строк/таблица» (≈ число позиций)**: если у marker меньше,
чем у Unlimited-OCR — он порезал таблицу. `num_tables > 1` для одной таблицы позиций = структура разорвана.

### REST API

```bash
# только Unlimited-OCR
curl -s -F file=@invoice.pdf -F mode=base http://localhost:7700/api/ocr | jq

# сравнение с готовым marker-выходом
curl -s -F file=@invoice.pdf -F mode=base -F marker_md=@invoice.marker.md \
  http://localhost:7700/api/compare | jq '{unlimited:.unlimited.stats, marker:.marker.stats}'
```

### CLI внутри контейнера (по SSH, без веба)

```bash
docker compose exec bench python -m scripts.compare_cli data/samples/invoice.pdf \
    --mode base --marker-md data/samples/invoice.marker.md
```

(положи тестовые файлы в `./data/samples/` на хосте — они видны внутри как `data/samples/`).

## Marker подпроцессом (опционально)

По умолчанию marker НЕ запускается (чтобы не делить VRAM с Unlimited-OCR) — сравнивай с готовым `.md`.
Чтобы запускать marker прямо в контейнере, раскомментируй `MARKER_CMD` в `docker-compose.yml`
(marker должен быть установлен в образе — добавь его в `requirements.txt`). `{input}`/`{output}` подставляются автоматически.

## SGLang (прод-альтернатива подачи, порт 7701)

```bash
bash scripts/serve_sglang.sh
```

OpenAI-совместимый сервер. На занятой GPU `--mem-fraction-static` уже выставлен в `0.5` (~16 GB),
иначе дефолтные 0.8 (~25.6 GB) + занятые 12 GB → OOM. Порт меняется через `SGLANG_PORT`.

## Без Docker (альтернатива)

```bash
bash setup.sh
source .venv/bin/activate
APP_PORT=7700 uvicorn app.main:app --host 0.0.0.0 --port 7700
```

## Заметка о достоверности

Стенд написан против документированного API модели, но `model.infer` / `model.infer_multi`
не прогонялись на этом железе. Обёртка читает markdown и из возвращаемого значения, и из
`output_path`, и логирует тип возврата (`raw_return_type`) — после первого прогона на сервере
при необходимости подправь `app/engines/unlimited.py` под фактический формат.
