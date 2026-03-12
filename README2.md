# RAG BIOS - Flujo del modelo

Este documento complementa el README principal y explica que parte del sistema corre localmente y que parte depende de OpenRouter.

## Vista general

La aplicacion tiene tres bloques principales:

- ingesta y preparacion de documentos
- recuperacion de evidencia
- generacion controlada de respuesta

## Flujo completo

```mermaid
flowchart TD
    A[Usuario carga archivos en Streamlit] --> B[Extraccion de texto]
    B --> C[Preparacion de documentos]
    C --> D[Embeddings locales]
    D --> E[Indice FAISS en memoria]
    F[Usuario hace una pregunta] --> G{Consulta tabular sobre XLSX?}
    E --> H[Retrieval semantico por distancia]
    G -- Si --> I[Lookup estructurado por fecha fila y columnas]
    G -- No --> H
    I --> J[Contexto final]
    H --> J
    J --> K[Prompt con evidencia etiquetada]
    K --> L[OpenRouter genera respuesta]
    L --> M[Validacion de citas]
    M --> N[Respuesta final y evidencia]
```

## Donde corre cada parte

```mermaid
flowchart LR
    subgraph Local[Equipo local]
        A1[Streamlit UI]
        A2[unstructured y openpyxl]
        A3[Chunking para PDF DOCX TXT]
        A4[Embeddings Hugging Face]
        A5[FAISS]
        A6[Lookup estructurado XLSX]
        A7[Seleccion por distancia]
        A8[Validacion de citas]
    end

    subgraph Remoto[OpenRouter]
        B1[LLM de generacion]
    end

    A1 --> A2 --> A3 --> A4 --> A5
    A1 --> A6
    A5 --> A7 --> B1
    A6 --> B1
    B1 --> A8 --> A1
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
    P->>P: Detecta si la pregunta parece tabular
    alt Pregunta tabular XLSX
        P->>P: Busca coincidencia estructurada por fecha fila y columnas
        P->>P: Si hay match exacto, construye respuesta o contexto directo
    else Pregunta general
        P->>F: similarity_search_with_score
        F-->>P: candidatos con distancia
        P->>P: ordena por distancia y recorta a TOP_K
    end
    P->>O: prompt + contexto final
    O-->>P: respuesta
    P->>P: valida citas y decide advertencia o abstencion
    P-->>S: respuesta final y evidencia
    S-->>U: muestra respuesta y fragmentos
```

## Control de alucinaciones

El flujo actual reduce alucinaciones con varias barreras:

- solo se usa contexto recuperado desde FAISS o desde lookup estructurado de XLSX
- el modelo recibe evidencia etiquetada como `[E1]`, `[E2]`, etc.
- solo se envian al modelo los mejores `TOP_K`
- para Excel, las preguntas exactas pueden resolverse sin depender del retrieval semantico
- si no hay evidencia suficiente, el sistema responde con abstencion
- si faltan citas validas, la aplicacion muestra advertencia y conserva la evidencia recuperada para inspeccion

## Flujo de decision de respuesta

```mermaid
flowchart TD
    A[Pregunta del usuario] --> B{Parece consulta tabular XLSX?}
    B -- Si --> C{Hay match estructurado?}
    C -- Si --> D[Construir respuesta o contexto desde la fila]
    C -- No --> E[Retrieval semantico por distancia]
    B -- No --> E
    E --> F{Hay evidencia recuperada?}
    F -- No --> G[Abstencion]
    F -- Si --> H[Construir contexto etiquetado]
    D --> I[Validar si requiere LLM]
    H --> J[Enviar al LLM]
    I --> J
    J --> K{La respuesta trae citas validas?}
    K -- Si --> L[Mostrar evidencia citada]
    K -- No --> M[Mostrar advertencia y evidencia recuperada]
    L --> N[Respuesta final]
    M --> N
```

## Costo y tokens en este flujo

- la sobre recuperacion no duplica por si sola el costo del LLM porque ocurre en FAISS local
- el costo remoto depende sobre todo de `TOP_K`, `CHUNK_SIZE` y del tamano real del contexto enviado
- el lookup estructurado de XLSX puede reducir costo porque evita mandar contexto innecesario al modelo
- las citas agregan pocos tokens extra en comparacion con enviar mas contexto

## Mapeo a archivos del proyecto

- `app.py`: interfaz Streamlit y render de respuesta y evidencia
- `src/rag_bios/document_loader.py`: extraccion de PDF, DOCX, XLSX y TXT
- `src/rag_bios/pipeline.py`: retrieval, lookup estructurado, armado de contexto y validacion de citas
- `src/rag_bios/prompts.py`: reglas del prompt grounded
- `src/rag_bios/config.py`: parametros de configuracion del flujo

## Como explicarlo en demo

Una forma simple de contarlo en la videollamada:

1. La aplicacion indexa documentos localmente con embeddings y FAISS.
2. Cuando el usuario pregunta, primero intenta recuperar evidencia local.
3. Si la pregunta es tabular sobre Excel, intenta resolverla con un camino estructurado mas preciso.
4. Solo despues llama al modelo remoto con contexto controlado, o responde directo si el lookup exacto ya resolvio el caso.
5. El LLM no es la fuente de verdad; la fuente de verdad son los documentos cargados y la evidencia mostrada en pantalla.
