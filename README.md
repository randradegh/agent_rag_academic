# Sistema de Evaluación de Ensayos Académicos

# 🦜🕸️LangGraph
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-32CD32.svg?logo=chainlink&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-1.3.0-412991.svg?logo=openai&logoColor=white)](https://openai.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.13-4B0082.svg)](https://python.langchain.com/docs/langgraph)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-FB7299.svg)](https://github.com/facebookresearch/faiss)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-154F3F.svg)](https://www.nltk.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org)

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

### Administración del Sistema
- Gestión dinámica de artículos de referencia
- Ocultamiento temporal de artículos no necesarios
- Depuración selectiva de evaluaciones
- Interfaz intuitiva para mantenimiento

### Sistema de Ayuda
- Guía detallada de uso del sistema
- Explicación de cada componente
- Flujo de trabajo recomendado
- Tips y mejores prácticas

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

1. **📄 Subir Artículos de Referencia**
   - Ingresar URL del artículo
   - El sistema procesará y almacenará el contenido

2. **📝 Evaluar Ensayos**
   - Subir el PDF del ensayo
   - El sistema identificará al estudiante
   - Se generará una evaluación detallada

3. **📊 Revisar Evaluaciones**
   - Acceder al historial de evaluaciones
   - Ver detalles y calificaciones

4. **🤖 Visualizar Flujo de Agentes**
   - Comprender el proceso interno
   - Ver la interacción entre agentes
   - Entender el flujo de trabajo

5. **⚙️ Administrar Sistema**
   - Gestionar artículos de referencia
   - Ocultar/mostrar artículos según necesidad
   - Depurar evaluaciones antiguas

6. **❓ Consultar Ayuda**
   - Acceder a la guía completa
   - Ver flujo de trabajo recomendado
   - Consultar tips y mejores prácticas

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