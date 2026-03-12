# RAG BIOS

Aplicacion web tipo chat para cargar documentos y hacer preguntas usando un flujo RAG con respuestas basadas unicamente en la evidencia recuperada.

## Objetivo

La solucion fue construida para cumplir la prueba tecnica de Analista IA:

- soporta PDF, DOCX, XLSX y TXT
- ofrece una interfaz simple tipo chat
- responde solo con base en el contenido cargado
- se abstiene o advierte cuando no hay suficiente soporte en la evidencia

## Arquitectura

El flujo general es:

1. El usuario carga uno o varios archivos desde Streamlit.
2. El sistema extrae texto de PDF, DOCX y TXT con `unstructured`.
3. Los archivos XLSX se procesan con `openpyxl` fila por fila y se enriquecen con metadata tabular.
4. PDF, DOCX y TXT se dividen en chunks con `RecursiveCharacterTextSplitter`.
5. Las filas de XLSX se mantienen intactas para no romper el contexto tabular.
6. El contenido se indexa en FAISS usando embeddings locales de Hugging Face.
7. Para cada pregunta, el sistema intenta primero un lookup estructurado si detecta una consulta tabular sobre Excel.
8. Si no aplica el lookup estructurado, usa retrieval semantico por distancia y envia solo los mejores `TOP_K` al modelo.
9. El modelo de OpenRouter responde con un prompt restrictivo que lo obliga a usar solo el contexto recuperado.
10. La interfaz muestra la respuesta y los fragmentos fuente usados como evidencia.

## Estructura del proyecto

```text
RAG_BIOS/
|-- app.py
|-- README.md
|-- README2.md
|-- pyproject.toml
|-- uv.lock
|-- .env.example
`-- src/
    `-- rag_bios/
        |-- __init__.py
        |-- config.py
        |-- document_loader.py
        |-- pipeline.py
        `-- prompts.py
```

`README2.md` complementa este documento con diagramas y una explicacion mas detallada del flujo.

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
- Si cambias dependencias, actualiza el lockfile con `uv lock`.

## Variables principales

- `OPENROUTER_API_KEY`: clave de acceso al modelo.
- `OPENROUTER_MODEL`: modelo a usar en OpenRouter.
- `OPENROUTER_BASE_URL`: endpoint del proveedor.
- `OPENROUTER_HTTP_REFERER`: cabecera usada por OpenRouter; en local puede apuntar a tu instancia de Streamlit.
- `OPENROUTER_X_TITLE`: titulo enviado al proveedor.
- `EMBEDDING_MODEL`: modelo local para embeddings.
- `CHUNK_SIZE`: tamano del chunk para indexacion.
- `CHUNK_OVERLAP`: solapamiento entre chunks.
- `TOP_K`: cantidad final de fragmentos que se envian al modelo en cada pregunta.
- `RETRIEVAL_MULTIPLIER`: cantidad de candidatos extra que se recuperan localmente antes de reordenar por distancia y recortar a `TOP_K`.
- `REQUIRE_CITATIONS`: intenta exigir citas como `[E1]`, `[E2]`; si faltan, la app muestra advertencia y evidencia recuperada.
- `TEMPERATURE`: temperatura del modelo; se recomienda `0.0` para reducir variabilidad.

## Decisiones tecnicas

- Se eligio Streamlit para acelerar la entrega y dejar una interfaz clara para demo.
- Se eligio OpenRouter como backend principal por flexibilidad de modelos.
- Los embeddings se calculan localmente para no depender de una segunda API.
- FAISS se mantiene en memoria durante la sesion para simplificar el MVP.
- XLSX se procesa aparte para representar cada fila como evidencia recuperable con hoja y numero de fila.
- Las preguntas exactas de Excel como `fecha + columna` o `fila + columna` usan un lookup estructurado primero, porque para tablas anchas es mas confiable que depender solo de embeddings.
- Las filas de Excel no se trocean en chunks para evitar perder relaciones entre columnas de una misma fila.
- La respuesta se restringe mediante prompt, evidencia recuperada visible en pantalla y advertencias si el modelo no cita bien.

## Como se reduce la alucinacion

- El prompt indica que la respuesta solo puede basarse en el contexto recuperado.
- El sistema recupera candidatos localmente, los ordena por distancia vectorial y solo envia al modelo los mejores `TOP_K`.
- En XLSX, antes del retrieval semantico se intenta resolver preguntas tabulares con un lookup estructurado sobre metadata de fila, fecha y columnas.
- Si el contexto recuperado no soporta la pregunta, el modelo debe responder: `No encontre esa informacion en los documentos cargados.`
- Si el modelo no devuelve citas validas, la app mantiene la evidencia recuperada visible y marca la respuesta con advertencia.
- La interfaz muestra los fragmentos recuperados para inspeccion manual.

## Costo y tokens

- `RETRIEVAL_MULTIPLIER` no incrementa directamente el costo de OpenRouter porque la sobre recuperacion y el reordenamiento ocurren en FAISS de forma local.
- El costo remoto lo dominan sobre todo `TOP_K`, `CHUNK_SIZE` y la longitud real de los fragmentos enviados al modelo.
- Las citas agregan un overhead pequeno de tokens porque la respuesta incluye etiquetas como `[E1]` y el contexto lleva identificadores de evidencia.
- En la configuracion actual, el sistema puede buscar mas candidatos localmente sin enviar todos esos candidatos al LLM.

## Ajuste recomendado de costo vs precision

- Si quieres bajar costo, reduce `TOP_K` antes de tocar `RETRIEVAL_MULTIPLIER`.
- Si quieres mantener precision sin disparar tokens, deja `TOP_K` bajo y usa `RETRIEVAL_MULTIPLIER` para refinar la seleccion local.
- Para Excel, antes de subir `TOP_K`, revisa si la pregunta puede expresarse como `fecha + columna`; ese camino suele ser mas preciso y mas barato que forzar mas contexto al LLM.
- Un punto de partida razonable para demo es `TOP_K=6`, `RETRIEVAL_MULTIPLIER=3`, `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200` y `TEMPERATURE=0.0`.

## Posibles mejoras futuras

- Persistencia del indice vectorial entre sesiones.
- Normalizacion adicional de encabezados y sinonimos para columnas de Excel.
- Ajustes especificos para TXT cortos o notas semi estructuradas.
- Metricas de evaluacion automatica para grounding.
- Citas mas precisas por pagina, celda o seccion.
- Soporte multiusuario.

## Guia corta para demo

1. Cargar un PDF o DOCX con informacion conocida.
2. Hacer una pregunta cuya respuesta este claramente en el documento.
3. Mostrar la respuesta y abrir los fragmentos fuente.
4. Hacer una pregunta fuera del contenido.
5. Mostrar que el sistema responde que no encontro la informacion.
6. Repetir con un XLSX para probar el cuarto formato exigido.
