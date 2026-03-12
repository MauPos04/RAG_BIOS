# RAG BIOS

Aplicacion web tipo chat para cargar documentos y hacer preguntas usando un flujo RAG con respuestas ancladas unicamente al contenido recuperado del documento.

## Objetivo

La solucion fue pensada para cumplir el reto tecnico de Analista IA:

- formatos soportados: PDF, DOCX, XLSX y TXT
- interfaz simple tipo chat
- respuestas generadas solo con base en el contenido cargado
- manejo explicito del caso sin evidencia para reducir alucinaciones

## Arquitectura

El flujo de la aplicacion es:

1. El usuario carga uno o varios archivos desde Streamlit.
2. El sistema extrae texto de PDF, DOCX y TXT con `unstructured`.
3. Los archivos XLSX se procesan hoja por hoja con `openpyxl` y se serializan a texto.
4. Los documentos se dividen en chunks con `RecursiveCharacterTextSplitter`.
5. Los chunks se indexan en FAISS usando embeddings locales de Hugging Face.
6. Para cada pregunta, se recuperan los fragmentos mas relevantes.
7. Se envia al modelo de OpenRouter un prompt restrictivo que obliga a responder solo con el contexto recuperado.
8. La interfaz muestra la respuesta y los fragmentos fuente usados como evidencia.

## Estructura del proyecto

```text
RAG_BIOS/
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
└── src/
    └── rag_bios/
        ├── __init__.py
        ├── config.py
        ├── document_loader.py
        ├── pipeline.py
        └── prompts.py
```

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
- `EMBEDDING_MODEL`: modelo local para embeddings.
- `CHUNK_SIZE`: tamano del chunk para indexacion.
- `CHUNK_OVERLAP`: solapamiento entre chunks.
- `TOP_K`: cantidad de fragmentos recuperados por pregunta.
- `MIN_RELEVANCE_SCORE`: umbral minimo para aceptar evidencia.
- `TEMPERATURE`: temperatura del modelo, recomendada en `0.0` para reducir variabilidad.

## Decisiones tecnicas

- Se eligio Streamlit para acelerar la entrega y dejar una interfaz clara para demo.
- Se eligio OpenRouter como backend principal por flexibilidad de modelos y costo.
- Los embeddings se calculan localmente para no depender de una segunda API.
- FAISS se mantiene en memoria durante la sesion para mejorar la experiencia frente a reconstruir el indice en cada pregunta.
- XLSX se procesa aparte para representar cada hoja y sus filas como texto recuperable.
- La respuesta se restringe mediante prompt y evidencia recuperada visible en pantalla.

## Como se reduce la alucinacion

- El prompt indica que la respuesta solo puede basarse en el contexto recuperado.
- Si no se supera el umbral de relevancia, la app responde: `No encontre esa informacion en los documentos cargados.`
- La interfaz muestra los fragmentos recuperados para inspeccion manual.
- La temperatura recomendada es `0.0`.

## Posibles mejoras futuras

- Persistencia del indice vectorial entre sesiones.
- Metricas de evaluacion automatica para grounding.
- Citas mas precisas por pagina, celda o seccion.
- Soporte multiusuario.
- Exportacion del historial de preguntas y respuestas.

## Guia corta para demo

1. Cargar un PDF o DOCX con informacion conocida.
2. Hacer una pregunta cuya respuesta este claramente en el documento.
3. Mostrar la respuesta y abrir los fragmentos fuente.
4. Hacer una pregunta fuera del contenido.
5. Mostrar que el sistema responde que no encontro la informacion.
6. Repetir con un XLSX para probar el cuarto formato exigido.