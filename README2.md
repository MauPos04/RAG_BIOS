# RAG BIOS - Flujo del modelo

Este documento complementa el README principal y describe que parte del sistema corre localmente y que parte depende de OpenRouter.

## Vista general

La aplicacion se organiza en tres bloques principales:

- ingesta y preparacion documental
- recuperacion de evidencia
- generacion controlada de respuesta

## Flujo completo

```mermaid
flowchart TD
    A[Usuario carga archivos en Streamlit] --> B[Extraccion y normalizacion]
    B --> C[Preparacion de documentos]
    C --> D[Embeddings locales]
    D --> E[Indice FAISS]
    F[Usuario hace una pregunta] --> G{La pregunta requiere ruta estructurada?}
    E --> H[Retrieval semantico general]
    G -- XLSX --> I[Lookup estructurado por fecha fila y columnas]
    G -- TXT --> J[Lookup estructurado por seccion]
    G -- No --> H
    I --> K[Contexto final]
    J --> K
    H --> K
    K --> L[Prompt con evidencia etiquetada]
    L --> M[OpenRouter genera respuesta]
    M --> N[Validacion de citas]
    N --> O[Respuesta final y evidencia]
```

## Donde corre cada parte

```mermaid
flowchart LR
    subgraph Local[Equipo local]
        A1[Streamlit UI]
        A2[unstructured y openpyxl]
        A3[Segmentacion y chunking]
        A4[Embeddings Hugging Face]
        A5[FAISS]
        A6[Lookup estructurado XLSX]
        A7[Lookup estructurado TXT]
        A8[Seleccion de evidencia]
        A9[Validacion de citas]
    end

    subgraph Remoto[OpenRouter]
        B1[LLM de generacion]
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
        P->>P: Busca coincidencia por seccion o comando
    else Consulta general
        P->>F: similarity_search_with_score
        F-->>P: candidatos con distancia
        P->>P: reordena y recorta a TOP_K
    end
    P->>O: prompt + contexto final
    O-->>P: respuesta
    P->>P: valida citas y decide advertencia o abstencion
    P-->>S: respuesta final y evidencia
    S-->>U: muestra respuesta y fragmentos
```

## Control de alucinaciones

El flujo actual reduce alucinaciones con varias barreras:

- solo se usa contexto recuperado desde FAISS o desde rutas estructuradas
- el modelo recibe evidencia etiquetada como `[E1]`, `[E2]`, etc.
- solo se envian al modelo los mejores `TOP_K`
- para `XLSX`, las preguntas exactas pueden resolverse sin depender del retrieval semantico
- para `TXT`, ciertas preguntas pueden resolverse desde la seccion correspondiente
- si no hay evidencia suficiente, el sistema responde con abstencion
- si faltan citas validas, la aplicacion conserva la evidencia recuperada para inspeccion

## Flujo de decision de respuesta

```mermaid
flowchart TD
    A[Pregunta del usuario] --> B{Requiere ruta estructurada?}
    B -- XLSX --> C{Hay match estructurado?}
    B -- TXT --> D{Hay match por seccion?}
    B -- No --> E[Retrieval semantico]
    C -- Si --> F[Construir respuesta o contexto desde la fila]
    C -- No --> E
    D -- Si --> G[Construir respuesta o contexto desde la seccion]
    D -- No --> E
    E --> H{Hay evidencia recuperada?}
    H -- No --> I[Abstencion]
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

- la sobre recuperacion no incrementa por si sola el costo del LLM porque ocurre en FAISS local
- el costo remoto depende sobre todo de `TOP_K`, `CHUNK_SIZE` y del tamano real del contexto enviado
- el lookup estructurado de `XLSX` puede reducir costo porque evita mandar contexto innecesario al modelo
- las citas agregan pocos tokens extra frente al costo de enviar mas contexto

## Mapeo a archivos del proyecto

- `app.py`: interfaz Streamlit, gestion de sesion y render de respuesta y evidencia
- `src/rag_bios/document_loader.py`: extraccion y normalizacion de `PDF`, `DOCX`, `XLSX` y `TXT`
- `src/rag_bios/pipeline.py`: retrieval, lookup estructurado, armado de contexto y validacion de citas
- `src/rag_bios/prompts.py`: reglas del prompt grounded
- `src/rag_bios/config.py`: parametros de configuracion del flujo
- `src/rag_bios/cache_store.py`: cache persistente del indice y documentos normalizados
