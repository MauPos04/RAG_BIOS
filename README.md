# RAG BIOS

Aplicacion web tipo chat para cargar documentos y realizar preguntas sobre su contenido mediante un flujo RAG con respuestas basadas unicamente en la evidencia recuperada.

## Objetivo

La solucion implementa un asistente RAG para consulta documental que:

- soporta `PDF`, `DOCX`, `XLSX` y `TXT`,
- ofrece una interfaz web simple tipo chat,
- responde solo con base en el contenido cargado,
- muestra evidencia visible para cada respuesta,
- se abstiene cuando no existe soporte suficiente en los documentos.

## Arquitectura

El flujo general es el siguiente:

1. El usuario carga uno o varios archivos desde Streamlit.
2. El sistema normaliza cada archivo segun su formato:
   - `PDF` y `DOCX`: extraccion textual con `unstructured`
   - `TXT`: segmentacion por secciones
   - `XLSX`: modelado por filas con metadata tabular
3. `PDF`, `DOCX` y `TXT` se fragmentan en chunks cuando corresponde.
4. Las filas de `XLSX` se mantienen intactas para preservar el contexto tabular.
5. El contenido se indexa en FAISS usando embeddings locales de Hugging Face.
6. Para cada pregunta, el sistema decide si aplica una ruta estructurada o retrieval semantico general:
   - lookup estructurado para consultas tabulares sobre `XLSX`
   - lookup estructurado por secciones para ciertas consultas sobre `TXT`
   - retrieval semantico general para preguntas sobre `PDF`, `DOCX` y consultas no estructuradas
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

Documentacion complementaria:

- `README2.md`: flujo del modelo y responsabilidades locales/remotas
- `COMO_FUNCIONA.md`: explicacion detallada del funcionamiento
- `DEMO_GUIA.md`: guion sugerido para la presentacion tecnica
- `PREGUNTAS_PRUEBA.md`: preguntas recomendadas para validacion funcional

## Instrucciones de ejecucion

1. Crear entorno virtual con Python 3.12 o superior.
2. Crear el entorno con `uv`:

```bash
uv venv
```

3. Activar el entorno virtual.

En Windows PowerShell:

```bash
.venv\Scripts\activate
```

4. Instalar dependencias bloqueadas por `uv.lock`:

```bash
uv sync
```

5. Copiar `.env.example` a `.env` y completar al menos `OPENROUTER_API_KEY`.
6. Ejecutar la aplicacion:

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
- `OPENROUTER_X_TITLE`: titulo enviado al proveedor.
- `EMBEDDING_MODEL`: modelo local para embeddings.
- `CHUNK_SIZE`: tamano del chunk para indexacion.
- `CHUNK_OVERLAP`: solapamiento entre chunks.
- `TOP_K`: cantidad final de fragmentos que se envian al modelo en cada pregunta.
- `RETRIEVAL_MULTIPLIER`: cantidad de candidatos extra recuperados localmente antes de recortar a `TOP_K`.
- `REQUIRE_CITATIONS`: habilita el control de citas como `[E1]`, `[E2]`.
- `TEMPERATURE`: temperatura del modelo.
- `DOCUMENT_LANGUAGES`: idiomas usados por `unstructured` para `PDF` y `DOCX`.
- `PERSISTENT_INDEX_CACHE`: activa cache persistente del procesamiento documental.
- `INDEX_CACHE_DIR`: directorio del cache persistente.
- `CHAT_MEMORY_TURNS`: cantidad de turnos recientes usados para follow-ups controlados.

## Decisiones tecnicas

- Se eligio Streamlit para ofrecer una interfaz web simple y mantenible.
- Se eligio OpenRouter como backend principal por flexibilidad en la seleccion del modelo.
- Los embeddings se calculan localmente para no depender de una segunda API remota.
- FAISS se usa como indice vectorial local para retrieval semantico.
- `XLSX` se procesa por fila para representar mejor la estructura tabular.
- Las preguntas tabulares exactas usan lookup estructurado antes del retrieval semantico.
- `TXT` se segmenta por secciones para preservar comandos y pasos operativos.
- La respuesta se restringe mediante prompt, evidencia visible y validacion posterior de citas.
- El sistema incorpora cache persistente opcional para reducir tiempos de reprocesamiento.

## Como se reduce la alucinacion

- El prompt indica que la respuesta solo puede basarse en el contexto recuperado.
- El sistema recupera evidencia localmente y solo envia al modelo los fragmentos mas relevantes.
- En `XLSX`, se intenta primero resolver preguntas tabulares exactas por metadata estructurada.
- En `TXT`, ciertas preguntas pueden resolverse a partir de secciones especificas.
- Si el contexto recuperado no soporta la pregunta, el sistema se abstiene.
- Si las citas no son validas, la interfaz conserva la evidencia recuperada para inspeccion.
- La evidencia utilizada se muestra en pantalla para auditoria manual.

## Costo y tokens

- La sobre recuperacion no incrementa directamente el costo del LLM porque ocurre en FAISS local.
- El costo remoto depende sobre todo de `TOP_K`, `CHUNK_SIZE` y del tamano real del contexto enviado.
- El lookup estructurado de `XLSX` puede reducir costo al evitar contexto innecesario.
- Las citas agregan un overhead pequeno de tokens en comparacion con enviar mas contexto.

## Ajuste recomendado de costo vs precision

- Para reducir costo, conviene ajustar `TOP_K` antes de modificar `RETRIEVAL_MULTIPLIER`.
- Para mantener precision sin aumentar tokens, conviene mantener `TOP_K` moderado y refinar la seleccion local.
- Para `XLSX`, antes de aumentar `TOP_K`, conviene reformular la pregunta en formato `fecha + columna`.
- Configuracion de referencia: `TOP_K=6`, `RETRIEVAL_MULTIPLIER=3`, `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` y `TEMPERATURE=0.0`.

## Posibles mejoras futuras

- Ajustes adicionales para preguntas ambiguas o de seguimiento complejo.
- Normalizacion adicional de encabezados y sinonimos para columnas de Excel.
- Metricas de evaluacion automatica para grounding.
- Citas mas precisas por pagina, celda o seccion.
- Mejor soporte para documentos escaneados.
- Soporte multiusuario.

## Guia de validacion funcional

1. Cargar un `PDF` o `DOCX` con informacion conocida.
2. Hacer una pregunta cuya respuesta este claramente en el documento.
3. Mostrar la respuesta y abrir los fragmentos fuente.
4. Hacer una pregunta fuera del contenido.
5. Mostrar que el sistema responde con abstencion controlada.
6. Repetir con un `XLSX` y un `TXT` para validar los cuatro formatos soportados.
