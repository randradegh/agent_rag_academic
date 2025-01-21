# Sistema de Evaluación de Ensayos Académicos

Este proyecto es una aplicación web desarrollada con Streamlit que utiliza Inteligencia Artificial para evaluar ensayos académicos, comparándolos con artículos de referencia. La aplicación proporciona una evaluación detallada y una calificación numérica basada en diversos criterios académicos.

## Características Principales

- **Gestión de Artículos de Referencia**
  - Carga de artículos web mediante URL
  - Almacenamiento persistente de artículos procesados
  - Sistema de vectores para análisis semántico

- **Evaluación de Ensayos**
  - Procesamiento automático de documentos PDF
  - Extracción automática del nombre del estudiante
  - Análisis detallado del contenido
  - Calificación numérica objetiva

- **Gestión de Evaluaciones**
  - Almacenamiento de resultados en formato CSV
  - Historial de evaluaciones realizadas
  - Interfaz intuitiva para visualizar resultados

## Tecnologías Utilizadas

- **Frontend**
  - Streamlit: Framework para la interfaz de usuario
  - Pandas: Manejo de datos tabulares

- **Procesamiento de Texto**
  - LangChain: Framework para aplicaciones de IA
  - FAISS: Biblioteca para búsqueda de similitud
  - NLTK: Procesamiento de lenguaje natural
  - BeautifulSoup4: Extracción de contenido web

- **Inteligencia Artificial**
  - OpenAI API: Modelos de lenguaje para análisis
  - Embeddings: Vectorización de texto

- **Almacenamiento**
  - Sistema de archivos local para vectorstores
  - CSV para historial de evaluaciones

## Requisitos

- Python 3.8 o superior
- API Key de OpenAI
- Dependencias listadas en `requirements.txt`

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd [NOMBRE_DEL_DIRECTORIO]
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno:
   - Crear archivo `.env`
   - Agregar API Key de OpenAI: `OPENAI_API_KEY=tu_api_key`

5. Ejecutar la aplicación:
```bash
streamlit run app.py
```

## Uso

1. **Subir Artículo de Referencia**
   - Acceder a la pestaña "Subir Artículo"
   - Ingresar URL del artículo de referencia
   - Procesar el artículo

2. **Evaluar Ensayos**
   - Ir a la pestaña "Evaluar Tarea"
   - Subir el PDF del ensayo a evaluar
   - Revisar la evaluación y calificación generada

## Estructura del Proyecto

```
.
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias del proyecto
├── .env               # Variables de entorno
├── vectorstores/      # Almacenamiento de vectores
└── evaluaciones.csv   # Historial de evaluaciones
```

## Características de la Evaluación

- **Análisis Detallado**
  - Resumen del ensayo
  - Relevancia del contenido
  - Coherencia y estructura
  - Uso de conceptos clave
  - Fortalezas y áreas de mejora

- **Calificación**
  - Escala de 0 a 10
  - Criterios objetivos de evaluación
  - Retroalimentación detallada

## Contribuir

Las contribuciones son bienvenidas. Por favor, siéntete libre de:
- Reportar bugs
- Sugerir nuevas características
- Enviar pull requests

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Autor

[Tu Nombre/Organización]

---
**Nota**: Este proyecto utiliza la API de OpenAI, asegúrate de tener una clave API válida y suficientes créditos para su uso. 