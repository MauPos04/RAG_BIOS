# RAG BIOS - Flujo del modelo

Este documento complementa el README principal y describe qué parte del sistema corre localmente y qué parte depende de OpenRouter.

## Vista general

La aplicación se organiza en tres bloques principales:

- ingesta y preparación documental
- recuperación de evidencia
- generación controlada de respuesta

## Flujo completo

```mermaid
flowchart TD
    A[Usuario carga archivos en Streamlit] --> B[Extracción y normalización]
    B --> C[Preparación de documentos]
    C --> D[Embeddings locales]
    D --> E[Índice FAISS]
    F[Usuario hace una pregunta] --> G{La pregunta requiere ruta estructurada?}
    E --> H[Retrieval semántico general]
    G -- XLSX --> I[Lookup estructurado por fecha fila y columnas]
    G -- TXT --> J[Lookup estructurado por sección]
    G -- No --> H
    I --> K[Contexto final]
    J --> K
    H --> K
    K --> L[Prompt con evidencia etiquetada]
    L --> M[OpenRouter genera respuesta]
    M --> N[Validación de citas]
    N --> O[Respuesta final y evidencia]
```

## Dónde corre cada parte

```mermaid
flowchart LR
    subgraph Local[Equipo local]
        A1[Streamlit UI]
        A2[unstructured y openpyxl]
        A3[Segmentación y chunking]
        A4[Embeddings Hugging Face]
        A5[FAISS]
        A6[Lookup estructurado XLSX]
        A7[Lookup estructurado TXT]
        A8[Selección de evidencia]
        A9[Validación de citas]
    end

    subgraph Remoto[OpenRouter]
        B1[LLM de generación]
    end

    A1 --> A2 --> A3 --> A4 --> A5
    A1 --> A6
    A1 --> A7
    A5 --> A8 --> B1
    A6 --> B1
    A7 --> B1
    B1 --> A9 --> A1
```

## Flujo por pregunta

```mermaid
sequenceDiagram
    participant U as Usuario
    participant S as Streamlit
    participant P as Pipeline local
    participant F as FAISS
    participant O as OpenRouter

    U->>S: Hace una pregunta
    S->>P: Ejecuta pipeline
    P->>P: Analiza si la consulta requiere ruta estructurada
    alt Consulta tabular XLSX
        P->>P: Busca coincidencia estructurada por fecha fila y columnas
        P->>P: Si hay match exacto, construye respuesta o contexto directo
    else Consulta por secciones TXT
        P->>P: Busca coincidencia por sección o comando
    else Consulta general
        P->>F: similarity_search_with_score
        F-->>P: candidatos con distancia
        P->>P: reordena y recorta a TOP_K
    end
    P->>O: prompt + contexto final
    O-->>P: respuesta
    P->>P: valida citas y decide advertencia o abstención
    P-->>S: respuesta final y evidencia
    S-->>U: muestra respuesta y fragmentos
```

## Control de alucinaciones

El flujo actual reduce alucinaciones con varias barreras:

- solo se usa contexto recuperado desde FAISS o desde rutas estructuradas
- el modelo recibe evidencia etiquetada como `[E1]`, `[E2]`, etc.
- solo se envían al modelo los mejores `TOP_K`
- para `XLSX`, las preguntas exactas pueden resolverse sin depender del retrieval semántico
- para `TXT`, ciertas preguntas pueden resolverse desde la sección correspondiente
- si no hay evidencia suficiente, el sistema responde con abstención
- si faltan citas válidas, la aplicación conserva la evidencia recuperada para inspección

## Flujo de decisión de respuesta

```mermaid
flowchart TD
    A[Pregunta del usuario] --> B{Requiere ruta estructurada?}
    B -- XLSX --> C{Hay match estructurado?}
    B -- TXT --> D{Hay match por sección?}
    B -- No --> E[Retrieval semántico]
    C -- Si --> F[Construir respuesta o contexto desde la fila]
    C -- No --> E
    D -- Si --> G[Construir respuesta o contexto desde la sección]
    D -- No --> E
    E --> H{Hay evidencia recuperada?}
    H -- No --> I[Abstención]
    H -- Si --> J[Construir contexto etiquetado]
    F --> K[Validar si requiere LLM]
    G --> K
    J --> L[Enviar al LLM]
    K --> L
    L --> M{La respuesta trae citas validas?}
    M -- Si --> N[Mostrar evidencia citada]
    M -- No --> O[Mostrar advertencia y evidencia recuperada]
    N --> P[Respuesta final]
    O --> P
```

## Costo y tokens en este flujo

- la sobre recuperación no incrementa por sí sola el costo del LLM porque ocurre en FAISS local
- el costo remoto depende sobre todo de `TOP_K`, `CHUNK_SIZE` y del tamaño real del contexto enviado
- el lookup estructurado de `XLSX` puede reducir costo porque evita mandar contexto innecesario al modelo
- las citas agregan pocos tokens extra frente al costo de enviar más contexto

## Mapeo a archivos del proyecto

- `app.py`: interfaz Streamlit, gestión de sesión y render de respuesta y evidencia
- `src/rag_bios/document_loader.py`: extracción y normalización de `PDF`, `DOCX`, `XLSX` y `TXT`
- `src/rag_bios/pipeline.py`: retrieval, lookup estructurado, armado de contexto y validación de citas
- `src/rag_bios/prompts.py`: reglas del prompt grounded
- `src/rag_bios/config.py`: parámetros de configuración del flujo
- `src/rag_bios/cache_store.py`: cache persistente del índice y documentos normalizados
