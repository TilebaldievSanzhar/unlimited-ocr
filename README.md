# unlimited-ocr-bench

Лёгкий стенд для сравнения **Baidu Unlimited-OCR** против **Marker** на твоих документах
(в первую очередь — инвойсы с длинными таблицами line items, где marker ломает структуру).

Сравнивается **только сырой OCR → markdown**. Этап классификации/экстракции здесь намеренно не трогаем.

## Что внутри

- `app/` — FastAPI сервис: веб-UI + REST API.
  - `engines/unlimited.py` — обёртка Unlimited-OCR (HF Transformers, `model.infer` / `model.infer_multi`).
  - `engines/marker.py` — опциональный запуск marker подпроцессом.
  - `pdf_utils.py` — PDF → PNG страницы (PyMuPDF, 300 DPI).
  - `metrics.py` — статистика по markdown-таблицам (число таблиц, строк, line items).
- `web/index.html` — ручное тестирование side-by-side.
- `scripts/compare_cli.py` — headless-прогон по SSH без веба.
- `scripts/serve_sglang.sh` — альтернативная подача через SGLang (OpenAI-совместимый API), для прода.
- `setup.sh` — установка окружения (важно: torch для Blackwell/sm_120).

## Требования

- Linux + NVIDIA GPU. На RTX 5090 (Blackwell, sm_120) **обязателен torch cu129** — иначе `no kernel image for sm_120`.
- Память: модель ~7.3 GB в bf16 + активации. На 5090 с ~20 GB свободными — с запасом. Квантизация не нужна.
- Python 3.12.

## Установка

```bash
git clone <your-remote> unlimited-ocr-bench
cd unlimited-ocr-bench
bash setup.sh
```

`setup.sh` ставит torch с индекса cu129 и остальные зависимости. Первый запуск скачает веса
`baidu/Unlimited-OCR` (несколько ГБ) в кэш HuggingFace.

> Если в `app/engines/unlimited.py` remote-код модели потребует `flash-attn`, поставь его отдельно:
> `pip install flash-attn --no-build-isolation` (сборка под Blackwell может занять время).

## Запуск веба

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Открой `http://<server>:8000/`. Загрузи PDF инвойса, выбери режим (`base` — точный, 1024px;
`gundam` — быстрый, 640px), при желании подгрузи `.md` из текущего marker — увидишь две колонки
с raw markdown и метриками. Ключевая метрика для инвойсов — **max строк в таблице** (≈ число позиций):
если у marker она меньше, чем у Unlimited-OCR, значит он порезал/сломал таблицу.

## REST API

```bash
# только Unlimited-OCR
curl -s -F file=@invoice.pdf -F mode=base http://localhost:8000/api/ocr | jq

# сравнение с готовым marker-выходом
curl -s -F file=@invoice.pdf -F mode=base -F marker_md=@invoice.marker.md \
  http://localhost:8000/api/compare | jq '{unlimited:.unlimited.stats, marker:.marker.stats}'
```

Полные markdown-выходы каждого прогона сохраняются в `data/outputs/<timestamp>/`.

## CLI (по SSH, без веба)

```bash
python -m scripts.compare_cli invoice.pdf --mode base --marker-md invoice.marker.md
```

## Marker подпроцессом (опционально)

По умолчанию marker в этом стенде НЕ запускается (чтобы не делить VRAM с Unlimited-OCR).
Если хочешь запускать его прямо здесь:

```bash
export MARKER_CMD="marker_single {input} --output_dir {output}"
```

`{input}` и `{output}` подставляются автоматически. Учти: marker догрузит свои модели на ту же GPU —
следи за памятью.

## SGLang (прод-альтернатива подачи)

```bash
bash scripts/serve_sglang.sh
```

Поднимает OpenAI-совместимый сервер. На занятой GPU **снизь `--mem-fraction-static`** (см. скрипт),
иначе OOM при 12 GB уже занятой памяти.

## Заметка о достоверности

Стенд написан против документированного API модели, но `model.infer` / `model.infer_multi`
не прогонялись на этом железе. Обёртка читает markdown и из возвращаемого значения, и из
`output_path` (на случай, если функция только пишет в файл) и логирует тип возврата — после
первого прогона на сервере при необходимости подправь `app/engines/unlimited.py` под фактический формат.
