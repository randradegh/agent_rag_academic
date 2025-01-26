# Sistema de Evaluación de Ensayos Académicos

Este proyecto es una aplicación web desarrollada con Streamlit que utiliza Inteligencia Artificial para evaluar ensayos académicos contra artículos de referencia, proporcionando evaluaciones detalladas y calificaciones numéricas.

## Características Principales

### Gestión de Artículos de Referencia
- Carga y almacenamiento de artículos académicos
- Análisis semántico de contenido
- Vectorización y almacenamiento eficiente

### Evaluación de Ensayos
- Procesamiento automático de PDFs
- Extracción inteligente de datos del estudiante
- Análisis de contenido y comparación con referencias
- Calificación numérica basada en criterios académicos

### Gestión de Evaluaciones
- Almacenamiento de resultados en CSV
- Historial de evaluaciones
- Interfaz intuitiva para revisión

## Arquitectura del Sistema

### Agentes de IA
El sistema utiliza una arquitectura basada en agentes de IA (Agentic AI), donde cada agente es un componente de software autónomo especializado:

1. **DocumentLoaderAgent**: Encargado de procesar y vectorizar los artículos de referencia
2. **TaskAnalyzerAgent**: Analiza y extrae información de los ensayos enviados
3. **EvaluationAgent**: Realiza la evaluación comparativa y genera calificaciones

### Flujo de Trabajo con LangGraph
El sistema implementa un flujo de trabajo orquestado por LangGraph, que permite:
- Ejecución secuencial y coordinada de agentes
- Gestión de estado entre componentes
- Control de flujo y manejo de errores

### Diagrama de Flujo
```
El sistema sigue un flujo de trabajo dirigido:

[__start__] -> [DocumentLoaderAgent] -> [TaskAnalyzerAgent] -> [EvaluationAgent] -> [__end__]
```

## Tecnologías Utilizadas

### Frontend
- Streamlit
- Pandas
- Graphviz (visualización de flujos)

### Procesamiento de Texto
- LangChain
- FAISS
- NLTK
- BeautifulSoup4

### IA y Embeddings
- OpenAI API
- LangGraph (orquestación de agentes)
- Embeddings para análisis semántico

### Almacenamiento
- Sistema de archivos local para vectorstores
- CSV para historial de evaluaciones

## Instalación

1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Configurar variables de entorno (API key de OpenAI)
4. Ejecutar: `streamlit run app.py`

## Uso

1. **Subir Artículos de Referencia**
   - Ingresar URL del artículo
   - El sistema procesará y almacenará el contenido

2. **Evaluar Ensayos**
   - Subir el PDF del ensayo
   - El sistema identificará al estudiante
   - Se generará una evaluación detallada

3. **Revisar Evaluaciones**
   - Acceder al historial de evaluaciones
   - Ver detalles y calificaciones

## Estructura del Proyecto
```
.
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias
├── vectorstores/      # Almacenamiento de vectores
└── evaluaciones.csv   # Historial de evaluaciones
```

## Características de Evaluación
- Análisis de similitud semántica
- Evaluación de estructura y coherencia
- Verificación de referencias y citas
- Calificación numérica objetiva

## Contribución
Las contribuciones son bienvenidas. Por favor, seguir el proceso estándar de fork y pull request.

## Licencia
Este proyecto está bajo la Licencia MIT.

## Autor
Idea y dirección de Roberto Andrade F. (@randradedev).
Con el apoyo de Cursor AI usando DeepSeek-R1.

**Nota**: Se requiere una API key válida de OpenAI para el funcionamiento del sistema. 