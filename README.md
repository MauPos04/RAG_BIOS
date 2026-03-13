# RAG BIOS

Aplicación web tipo chat para cargar documentos y realizar preguntas sobre su contenido mediante un flujo RAG con respuestas basadas únicamente en la evidencia recuperada.

## Objetivo

La solución implementa un asistente RAG para consulta documental que:

- soporta `PDF`, `DOCX`, `XLSX` y `TXT`,
- ofrece una interfaz web simple tipo chat,
- responde solo con base en el contenido cargado,
- muestra evidencia visible para cada respuesta,
- se abstiene cuando no existe soporte suficiente en los documentos.

## Arquitectura

El flujo general es el siguiente:

1. El usuario carga uno o varios archivos desde Streamlit.
2. El sistema normaliza cada archivo según su formato:
   - `PDF` y `DOCX`: extracción textual con `unstructured`
   - `TXT`: segmentación por secciones
   - `XLSX`: modelado por filas con metadata tabular
3. `PDF`, `DOCX` y `TXT` se fragmentan en chunks cuando corresponde.
4. Las filas de `XLSX` se mantienen intactas para preservar el contexto tabular.
5. El contenido se indexa en FAISS usando embeddings locales de Hugging Face.
6. Para cada pregunta, el sistema decide si aplica una ruta estructurada o retrieval semántico general:
   - lookup estructurado para consultas tabulares sobre `XLSX`
   - lookup estructurado por secciones para ciertas consultas sobre `TXT`
   - retrieval semántico general para preguntas sobre `PDF`, `DOCX` y consultas no estructuradas
7. El modelo de OpenRouter genera la respuesta usando solo el contexto recuperado.
8. La respuesta se valida, se citan evidencias y la interfaz muestra los fragmentos fuente utilizados.

## Estructura del proyecto

```text
RAG_BIOS/
|-- app.py
|-- README.md
|-- README2.md
|-- COMO_FUNCIONA.md
|-- DEMO_GUIA.md
|-- PREGUNTAS_PRUEBA.md
|-- pyproject.toml
|-- uv.lock
|-- .env.example
`-- src/
    `-- rag_bios/
        |-- __init__.py
        |-- cache_store.py
        |-- config.py
        |-- document_loader.py
        |-- pipeline.py
        `-- prompts.py
```

## Instrucciones de ejecución

1. Crear entorno virtual con Python 3.12 o superior.
2. Crear el entorno con `uv`:

```bash
uv venv
```

3. Activar el entorno virtual según el sistema operativo.

En Windows PowerShell:

```bash
.venv\Scripts\activate
```

En macOS:

```bash
source .venv/bin/activate
```

En Linux:

```bash
source .venv/bin/activate
```

4. Instalar dependencias bloqueadas por `uv.lock`:

```bash
uv sync
```

5. Copiar `.env.example` a `.env` y completar al menos `OPENROUTER_API_KEY`.
6. Ejecutar la aplicación:

```bash
streamlit run app.py
```

## Flujo con uv

- `pyproject.toml` define las dependencias del proyecto.
- `uv.lock` fija las versiones exactas para mantener un entorno reproducible.
- `uv sync` instala exactamente lo definido en el lockfile.
- Si cambian dependencias, el lockfile puede actualizarse con `uv lock`.

## Variables principales

- `OPENROUTER_API_KEY`: clave de acceso al modelo.
- `OPENROUTER_MODEL`: modelo a usar en OpenRouter.
- `OPENROUTER_BASE_URL`: endpoint del proveedor.
- `OPENROUTER_HTTP_REFERER`: cabecera usada por OpenRouter.
- `OPENROUTER_X_TITLE`: título enviado al proveedor.
- `EMBEDDING_MODEL`: modelo local para embeddings.
- `CHUNK_SIZE`: tamaño del chunk para indexación.
- `CHUNK_OVERLAP`: solapamiento entre chunks.
- `TOP_K`: cantidad final de fragmentos que se envían al modelo en cada pregunta.
- `RETRIEVAL_MULTIPLIER`: cantidad de candidatos extra recuperados localmente antes de recortar a `TOP_K`.
- `REQUIRE_CITATIONS`: habilita el control de citas como `[E1]`, `[E2]`.
- `TEMPERATURE`: temperatura del modelo.
- `DOCUMENT_LANGUAGES`: idiomas usados por `unstructured` para `PDF` y `DOCX`.
- `PERSISTENT_INDEX_CACHE`: activa cache persistente del procesamiento documental.
- `INDEX_CACHE_DIR`: directorio del cache persistente.
- `CHAT_MEMORY_TURNS`: cantidad de turnos recientes usados para follow-ups controlados.

## Decisiones técnicas

- Se eligió Streamlit para ofrecer una interfaz web simple y mantenible.
- Se eligió OpenRouter como backend principal por flexibilidad en la selección del modelo.
- Los embeddings se calculan localmente para no depender de una segunda API remota.
- FAISS se usa como índice vectorial local para retrieval semántico.
- `XLSX` se procesa por fila para representar mejor la estructura tabular.
- Las preguntas tabulares exactas usan lookup estructurado antes del retrieval semántico.
- `TXT` se segmenta por secciones para preservar comandos y pasos operativos.
- La respuesta se restringe mediante prompt, evidencia visible y validación posterior de citas.
- El sistema incorpora cache persistente opcional para reducir tiempos de reprocesamiento.

## Cómo se reduce la alucinación

- El prompt indica que la respuesta solo puede basarse en el contexto recuperado.
- El sistema recupera evidencia localmente y solo envía al modelo los fragmentos más relevantes.
- En `XLSX`, se intenta primero resolver preguntas tabulares exactas por metadata estructurada.
- En `TXT`, ciertas preguntas pueden resolverse a partir de secciones específicas.
- Si el contexto recuperado no soporta la pregunta, el sistema se abstiene.
- Si las citas no son válidas, la interfaz conserva la evidencia recuperada para inspección.
- La evidencia utilizada se muestra en pantalla para auditoría manual.

## Costo y tokens

- La sobre recuperación no incrementa directamente el costo del LLM porque ocurre en FAISS local.
- El costo remoto depende sobre todo de `TOP_K`, `CHUNK_SIZE` y del tamaño real del contexto enviado.
- El lookup estructurado de `XLSX` puede reducir costo al evitar contexto innecesario.
- Las citas agregan un overhead pequeño de tokens en comparación con enviar más contexto.

## Ajuste recomendado de costo vs precisión

- Para reducir costo, conviene ajustar `TOP_K` antes de modificar `RETRIEVAL_MULTIPLIER`.
- Para mantener precisión sin aumentar tokens, conviene mantener `TOP_K` moderado y refinar la selección local.
- Para `XLSX`, antes de aumentar `TOP_K`, conviene reformular la pregunta en formato `fecha + columna`.
- Configuración de referencia: `TOP_K=6`, `RETRIEVAL_MULTIPLIER=3`, `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` y `TEMPERATURE=0.0`.

## Posibles mejoras futuras

- Ajustes adicionales para preguntas ambiguas o de seguimiento complejo.
- Normalización adicional de encabezados y sinónimos para columnas de Excel.
- Métricas de evaluación automática para grounding.
- Citas más precisas por página, celda o sección.
- Mejor soporte para documentos escaneados.
- Soporte multiusuario.

## Guía de validación funcional

1. Cargar un `PDF` o `DOCX` con información conocida.
2. Hacer una pregunta cuya respuesta esté claramente en el documento.
3. Mostrar la respuesta y abrir los fragmentos fuente.
4. Hacer una pregunta fuera del contenido.
5. Mostrar que el sistema responde con abstención controlada.
6. Repetir con un `XLSX` y un `TXT` para validar los cuatro formatos soportados.
