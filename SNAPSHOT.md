# Resona — Snapshot `v0.1.0-prealpha.1`

Congela la versión **usable** del Resona previa a la migración a la arquitectura de tres planos (presentación / lógica / engine).

## Usable en este punto

- **Studio** (`resona studio`): diseño, clonación y blend de voces, previews, guardado en biblioteca.
- **Biblioteca y búsqueda semántica** (pestañas Library / Search del Studio).
- **Emotion CAD** (`resona emotions`): catálogo de emociones/intensificadores con localización en 10 idiomas, CRUD y mini-testers.
- **DialogSpace** (`resona dialogs`): proyectos multi-voz con parámetros acústicos por línea, mastering global y compilación final (WAV/MP3).
- **AudioBooks**: compilación `.m4b` con capítulos, metadatos y carátula.
- **Voice API** (`resona voice-api`): voces, búsqueda semántica, emociones, síntesis asíncrona y diálogos.
- **CLI**: `resona studio | emotions | dialogs | voice-api`.

## Roto / incompleto en este punto

- `src/tune/` (fine-tuning): imports rotos (`from config import settings`), sin activación de modelos afinados, sin UI ni API. **No usar.**
- `GET /v1/voices/{id}/bundle` del API: llama `pack_bundle` con firma desactualizada. Usar el export `.rnb` de la UI.
- `tests/test_audio_engine.py`: desactualizado (falta el argumento `mastering` en `merge_segments`).
- Varias apps sobre la misma GPU compiten por VRAM (sin arbitraje de cola).

## Requisitos de arranque

- Python 3.12 + `uv sync`.
- Modelos Qwen3-TTS pre-descargados en la caché de Hugging Face (el engine usa `local_files_only=True`).
- `ffmpeg` en el sistema.

## Datos locales (NO versionados)

`data/`, `bookspace/`, `dialogspace/`, `emotionspace/`, `apispace/`, `tunespace/` y `temp_poc_emotions/` quedan fuera del repo: un clon arranca con base de datos vacía.

- Voces: exportar/importar bundles `.rnb`.
- Proyectos de diálogo y audiolibros: copiar sus directorios manualmente.
