import streamlit as st
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# Configuración de la página (debe ser lo primero)
st.set_page_config(
    page_title="Evaluador de Ensayos",
    layout="wide",
    initial_sidebar_state="expanded"
)

from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from operator import itemgetter
import os
from dotenv import load_dotenv
import nltk
import magic
import requests
from bs4 import BeautifulSoup
import time
import tempfile
import hashlib
import pandas as pd
from datetime import datetime
import re
from urllib.parse import urlparse

# Configuración de temas
# Tema claro (default)
light_theme = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#31333F",
    "font": "sans serif"
}

# Tema oscuro
dark_theme = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#121212",
    "secondaryBackgroundColor": "#1F1F1F",
    "textColor": "#FFFFFF",
    "font": "sans serif"
}

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI API Key desde .env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.environ.get("OPENAI_API_KEY"):
    st.error("No se encontró la API Key de OpenAI en el archivo .env")
    st.stop()

# Configurar directorio para almacenar vectorstores
VECTOR_STORE_DIR = "vectorstores"
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

# Configurar directorio de datos NLTK
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def extraer_titulo_articulo(url):
    """Extrae el título o primer H1 de un artículo"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Intentar obtener el título del artículo
        # Primero buscar en h1
        h1 = soup.find('h1')
        if h1 and h1.text.strip():
            return h1.text.strip()
        
        # Si no hay h1, intentar con el título de la página
        title = soup.find('title')
        if title and title.text.strip():
            return title.text.strip()
            
        return "Sin título"
    except Exception:
        return "Sin título"

def listar_articulos_guardados():
    """Lista todos los artículos guardados"""
    articulos = []
    for archivo in os.listdir(VECTOR_STORE_DIR):
        if archivo.endswith("_meta.txt"):
            vector_id = archivo.replace("_meta.txt", "")
            with open(os.path.join(VECTOR_STORE_DIR, archivo), "r") as f:
                url = f.read().strip()
            # Intentar obtener el título del artículo
            titulo = extraer_titulo_articulo(url)
            articulos.append({
                'id': vector_id,
                'url': url,
                'titulo': titulo
            })
    return articulos

def cargar_vectorstore(vector_id):
    """Carga un vectorstore desde disco"""
    vector_path = os.path.join(VECTOR_STORE_DIR, vector_id)
    if os.path.exists(vector_path):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    return None

# Selector de tema en la barra lateral
with st.sidebar:
    st.title("Configuración")
    theme = st.selectbox(
        "Tema",
        options=["Claro", "Oscuro"],
        index=0
    )
    
    # Aplicar tema seleccionado
    if theme == "Oscuro":
        st.markdown("""
            <style>
                .stApp {
                    background-color: #121212;
                }
                .stMarkdown, .stText, .stTitle, div[data-testid="stMarkdownContainer"] {
                    color: white !important;
                }
                .stButton button {
                    background-color: #1F1F1F !important;
                    color: white !important;
                    border: 1px solid #333 !important;
                    font-size: 14px !important;
                    padding: 4px 12px !important;
                }
                .stButton button:hover {
                    border-color: #666 !important;
                    background-color: #333 !important;
                }
                .stTextInput input, .stSelectbox select, div[data-testid="stFileUploader"] {
                    background-color: #1F1F1F !important;
                    color: white !important;
                    border: 1px solid #333 !important;
                }
                .stDataFrame {
                    background-color: #1F1F1F !important;
                }
                div.stTabs button {
                    background-color: #1F1F1F !important;
                    color: white !important;
                }
                div.stTabs button[data-baseweb="tab"] {
                    border-color: #333 !important;
                }
                div.stTabs button[aria-selected="true"] {
                    background-color: #333 !important;
                }
                .stSidebar {
                    background-color: #1F1F1F !important;
                }
                .stSidebar .stMarkdown, 
                .stSidebar .stTitle, 
                .stSidebar [data-testid="stMarkdownContainer"] {
                    color: white !important;
                }
                .stSidebar .stSelectbox label {
                    color: white !important;
                }
                .article-title {
                    font-size: 0.8em;
                    margin-top: -0.5em;
                    margin-bottom: 1em;
                    color: #666;
                }
                /* Estilos para la tabla en modo oscuro */
                [data-testid="stDataFrame"] table {
                    border: 1px solid #333 !important;
                }
                [data-testid="stDataFrame"] th {
                    border-bottom: 2px solid #333 !important;
                    background-color: #1F1F1F !important;
                    color: white !important;
                }
                [data-testid="stDataFrame"] td {
                    border: 1px solid #333 !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                .stApp {
                    background-color: #FFFFFF;
                }
                .stMarkdown, .stText, .stTitle, div[data-testid="stMarkdownContainer"] {
                    color: #31333F !important;
                }
                .stButton button {
                    background-color: #F0F2F6 !important;
                    color: #31333F !important;
                    border: 1px solid #ddd !important;
                    font-size: 14px !important;
                    padding: 4px 12px !important;
                }
                .stButton button:hover {
                    border-color: #999 !important;
                    background-color: #E8EAF1 !important;
                }
                /* Estilos para el tooltip en modo claro */
                button[data-baseweb="button"] {
                    position: relative;
                }
                button[data-baseweb="button"]:hover::before {
                    content: attr(aria-label);
                    position: absolute;
                    bottom: 100%;
                    left: 50%;
                    transform: translateX(-50%);
                    padding: 4px 8px;
                    background-color: rgba(0, 0, 0, 0.8);
                    color: white;
                    border-radius: 4px;
                    font-size: 12px;
                    white-space: nowrap;
                    z-index: 1000;
                    text-shadow: none;
                    font-weight: normal;
                    letter-spacing: normal;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }
                /* Flecha del tooltip */
                button[data-baseweb="button"]:hover::after {
                    content: "";
                    position: absolute;
                    bottom: 90%;
                    left: 50%;
                    transform: translateX(-50%);
                    border-width: 5px;
                    border-style: solid;
                    border-color: #FFFFFF transparent transparent transparent;
                    filter: drop-shadow(0 1px 1px rgba(0,0,0,0.1));
                    z-index: 1000;
                }
                /* Estilos para la tabla en modo claro */
                [data-testid="stDataFrame"] {
                    background-color: #FFFFFF;
                }
                [data-testid="stDataFrame"] table {
                    border: 1px solid #ccc !important;
                }
                [data-testid="stDataFrame"] th {
                    background-color: #F0F2F6 !important;
                    color: #31333F !important;
                    border: 1px solid #ccc !important;
                    border-bottom: 2px solid #999 !important;
                    font-weight: 600 !important;
                }
                [data-testid="stDataFrame"] td {
                    border: 1px solid #ccc !important;
                    color: #31333F !important;
                }
                [data-testid="stDataFrame"] tr:hover {
                    background-color: #f8f9fa !important;
                }
                /* Estilos para el contenedor de detalles */
                .detalle-evaluacion {
                    padding: 1.5rem !important;
                    border-radius: 0.5rem !important;
                    border: 1px solid #ccc !important;
                    margin: 1rem 0 !important;
                    background-color: #FFFFFF !important;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
                }
                /* Otros estilos del modo claro */
                .stTextInput input, .stSelectbox select, div[data-testid="stFileUploader"] {
                    background-color: #FFFFFF !important;
                    color: #31333F !important;
                    border: 1px solid #ddd !important;
                }
                .stDataFrame {
                    background-color: #FFFFFF !important;
                }
                div.stTabs button {
                    background-color: #FFFFFF !important;
                    color: #31333F !important;
                }
                div.stTabs button[data-baseweb="tab"] {
                    border-color: #ddd !important;
                }
                div.stTabs button[aria-selected="true"] {
                    background-color: #F0F2F6 !important;
                }
                .stSidebar {
                    background-color: #F0F2F6 !important;
                }
                .stSidebar .stMarkdown, 
                .stSidebar .stTitle, 
                .stSidebar [data-testid="stMarkdownContainer"] {
                    color: #31333F !important;
                }
                .stSidebar .stSelectbox label {
                    color: #31333F !important;
                }
            </style>
        """, unsafe_allow_html=True)
    
    # Mostrar artículos guardados en el sidebar
    st.divider()
    st.subheader("Artículos Guardados")
    if 'articulos_procesados' in st.session_state and st.session_state.articulos_procesados:
        # Selector múltiple de artículos
        opciones_articulos = [f"{art['titulo']} ({art['url']})" for art in st.session_state.articulos_procesados]
        seleccion_articulos = st.multiselect(
            "Selecciona artículos para evaluación:",
            opciones_articulos,
            key="selector_articulos",
            on_change=lambda: st.session_state.update({'vectorstore': None})  # Resetear al cambiar selección
        )
        
        # Actualizar vectorstore con los seleccionados
        if seleccion_articulos:
            vectorstores = []
            for art in st.session_state.articulos_procesados:
                if f"{art['titulo']} ({art['url']})" in seleccion_articulos:
                    vs = cargar_vectorstore(art['id'])
                    if vs:
                        vectorstores.append(vs)
            
            if vectorstores:
                # Obtener embeddings del primer vectorstore
                embeddings = vectorstores[0].embeddings
                
                # Combinar todos los documentos de forma segura
                docs_combinados = []
                for vs in vectorstores:
                    if hasattr(vs, 'docstore') and hasattr(vs.docstore, '_dict'):
                        documentos = vs.docstore._dict.values()
                        if documentos:
                            docs_combinados.extend([
                                doc for doc in documentos 
                                if hasattr(doc, 'page_content') and doc.page_content.strip()
                            ])
                
                if not docs_combinados:
                    st.error("No hay contenido válido en los artículos seleccionados")
                    st.session_state['vectorstore'] = None
                else:
                    # Crear nuevo vectorstore con los documentos combinados
                    combined_vs = FAISS.from_documents(
                        documents=docs_combinados,
                        embedding=embeddings
                    )
                    st.session_state['vectorstore'] = combined_vs
    else:
        st.info("No hay artículos guardados")

# Título principal
st.title("Sistema de Evaluación de Ensayos")

# Instrucciones del proceso
st.markdown("""
### Instrucciones del Proceso
1. **Paso 1:** En la pestaña 'Subir Artículo', ingresa la URL del artículo de referencia.
2. **Paso 2:** En la pestaña 'Evaluar Tarea', sube el archivo PDF con la tarea del estudiante.
3. **Paso 3:** El sistema evaluará automáticamente la tarea y proporcionará una calificación detallada.

**Nota:** Asegúrate de procesar primero el artículo de referencia antes de subir la tarea.
""")

# Lista actualizada de recursos NLTK necesarios
recursos_nltk = [
    'punkt',
    'averaged_perceptron_tagger',
    'words'
]

def descargar_recursos_nltk():
    for recurso in recursos_nltk:
        try:
            nltk.download(recurso, download_dir=nltk_data_dir, quiet=True)
            time.sleep(1)  # Pequeña pausa entre descargas
        except Exception as e:
            st.warning(f"No se pudo descargar el recurso {recurso}: {str(e)}")

# Intentar descargar recursos
try:
    descargar_recursos_nltk()
except Exception as e:
    st.error(f"Error al descargar recursos NLTK: {str(e)}")

# Función para validar URL
def es_url_valida(url):
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        return requests.head(url, timeout=5).status_code == 200
    except:
        return False

# Función para extraer texto de URL usando BeautifulSoup como respaldo
def extraer_texto_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        # Remover scripts, estilos y otros elementos no deseados
        for script in soup(["script", "style", "meta", "noscript", "header", "footer"]):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        raise Exception(f"Error al extraer texto de la URL: {str(e)}")

def generar_id_url(url):
    """Genera un ID único para la URL"""
    return hashlib.md5(url.encode()).hexdigest()

def guardar_vectorstore(vectorstore, url):
    """Guarda el vectorstore en disco"""
    vector_id = generar_id_url(url)
    vector_path = os.path.join(VECTOR_STORE_DIR, vector_id)
    vectorstore.save_local(vector_path)
    
    # Guardar metadata en JSON
    metadata = {
        'id': vector_id,
        'url': url,
        'titulo': extraer_titulo_articulo(url),
        'fecha_procesado': datetime.now().isoformat()
    }
    with open(os.path.join(VECTOR_STORE_DIR, f"{vector_id}.json"), 'w') as f:
        json.dump(metadata, f)
    
    return vector_id

# Cargar artículos al inicio
if 'articulos_procesados' not in st.session_state:
    st.session_state['articulos_procesados'] = []
    if os.path.exists(VECTOR_STORE_DIR):
        for file in os.listdir(VECTOR_STORE_DIR):
            if file.endswith('.json'):
                with open(os.path.join(VECTOR_STORE_DIR, file), 'r') as f:
                    metadata = json.load(f)
                    # Verificar que no exista ya en la lista usando ID
                    if not any(art['id'] == metadata['id'] for art in st.session_state.articulos_procesados):
                        st.session_state['articulos_procesados'].append(metadata)
    # Forzar actualización del sidebar
    st.rerun()

# Función para procesar URL y crear base de vectores
def procesar_articulo(url):
    if not es_url_valida(url):
        raise ValueError("La URL proporcionada no es válida o no está accesible")
    
    try:
        # Primer intento con UnstructuredURLLoader
        loader = UnstructuredURLLoader(urls=[url])
        data = loader.load()
        
        # Si no hay datos, intentar con BeautifulSoup
        if not data:
            texto = extraer_texto_url(url)
            if not texto:
                raise ValueError("No se pudo extraer contenido de la URL")
            data = [Document(page_content=texto, metadata={"source": url})]
        
        # Dividir el texto en chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
        
        if not docs:
            raise ValueError("No se pudo procesar el contenido del artículo")
        
        # Crear embeddings y almacenar en FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Guardar en disco
        vector_id = guardar_vectorstore(vectorstore, url)
        return vectorstore, vector_id
        
    except Exception as e:
        st.error(f"Error al procesar el artículo: {str(e)}")
        return None, None

def obtener_calificacion_numerica(texto):
    """Extrae un número válido de un texto, devolviendo 7.0 si no encuentra ninguno"""
    # Eliminar todo excepto números y puntos
    numeros = ''.join(c for c in texto if c.isdigit() or c == '.')
    
    try:
        # Intentar convertir a float
        if numeros:
            valor = float(numeros)
            # Validar que esté en el rango correcto
            if 0 <= valor <= 10:
                return round(valor, 1)
    except:
        pass
    
    return 7.0

def evaluar_ensayo(ensayo, vectorstores, nombre_alumno):
    """Evalúa el ensayo usando múltiples artículos de referencia"""
    try:
        # Combinar todos los documentos de referencia de forma segura
        docs_combinados = []
        for vs in vectorstores:
            if hasattr(vs, 'docstore') and hasattr(vs.docstore, '_dict'):
                documentos = vs.docstore._dict.values()
                if documentos:
                    docs_combinados.extend([
                        doc for doc in documentos 
                        if hasattr(doc, 'page_content') and doc.page_content.strip()
                    ])
        
        if not docs_combinados:
            raise ValueError("No se encontraron documentos válidos para comparar")
        
        # Crear contexto combinado con máximo 15,000 tokens aprox
        contexto = "\n\n".join([doc.page_content[:2000] for doc in docs_combinados if doc.page_content.strip()])[:15000]
        
        if not contexto.strip():
            raise ValueError("El contexto combinado está vacío")
        
        # Primer prompt para el análisis cualitativo
        prompt_analisis = f"""
        Eres un profesor experto evaluando ensayos académicos. Analiza el siguiente ensayo del estudiante {nombre_alumno}, 
        comparándolo con los siguientes artículos de referencia:
        
        {contexto}
        
        Estructura tu análisis de la siguiente manera:
        
        1. RESUMEN DEL ENSAYO (2-3 párrafos)
        - Principales ideas y argumentos presentados
        - Objetivo del ensayo
        
        2. ANÁLISIS DETALLADO
        a) Relevancia del contenido
        - Conexión con el artículo de referencia
        - Comprensión del tema
        - Uso de ejemplos y evidencias
        
        b) Coherencia y estructura
        - Organización lógica de ideas
        - Transiciones entre párrafos
        - Claridad en la argumentación
        
        c) Uso de conceptos clave
        - Identificación de términos importantes
        - Aplicación correcta de conceptos
        - Profundidad en el manejo de ideas
        
        3. FORTALEZAS Y ÁREAS DE MEJORA
        - Aspectos destacables del ensayo
        - Recomendaciones específicas para mejorar
        
        Ensayo a evaluar: {ensayo}
        """
        
        # Realizar el análisis cualitativo
        llm = ChatOpenAI(temperature=0.3)
        prompt = PromptTemplate(template=prompt_analisis, input_variables=["ensayo", "nombre_alumno"])
        chain = prompt | llm
        
        resultado = chain.invoke({"ensayo": ensayo, "nombre_alumno": nombre_alumno})
        analisis = resultado.content if hasattr(resultado, 'content') else str(resultado)
        
        # Segundo prompt específico para la calificación
        prompt_calificacion = """
        Basándote en el siguiente análisis de un ensayo, asigna una calificación numérica entre 0 y 10.
        RESPONDE ÚNICAMENTE CON EL NÚMERO, sin texto adicional.

        Análisis del ensayo:
        {analisis}
        """
        
        # Obtener la calificación numérica
        prompt_calif = PromptTemplate(template=prompt_calificacion, input_variables=["analisis"])
        chain_calif = prompt_calif | llm
        
        resultado_calif = chain_calif.invoke({"analisis": analisis})
        calif_texto = resultado_calif.content if hasattr(resultado_calif, 'content') else str(resultado_calif)
        
        try:
            # Limpiar y convertir a número
            calif_texto_limpio = ''.join(c for c in calif_texto if c.isdigit() or c == '.')
            calificacion = float(calif_texto_limpio)
            
            # Asegurar que está en el rango correcto
            calificacion = max(0.0, min(10.0, calificacion))
            calificacion = round(calificacion, 1)
        except:
            calificacion = 7.0
        
        # Agregar la calificación al análisis
        resultado_final = f"{analisis}\n\nCALIFICACIÓN FINAL: {calificacion}"
        
        return resultado_final, calificacion
        
    except Exception as e:
        return "No se pudo completar la evaluación. Se asignará una calificación por defecto.", 7.0

def guardar_evaluacion(nombre_alumno, calificacion, evaluacion_completa):
    """Guarda la evaluación en el CSV actualizado"""
    try:
        df = cargar_evaluaciones()
        nueva_evaluacion = {
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'nombre_alumno': nombre_alumno,
            'calificacion': calificacion,
            'evaluacion_completa': evaluacion_completa
        }
        df = pd.concat([df, pd.DataFrame([nueva_evaluacion])], ignore_index=True)
        df.to_csv("evaluaciones.csv", index=False)
        return df
    except Exception as e:
        return pd.DataFrame(columns=['fecha', 'nombre_alumno', 'calificacion', 'evaluacion_completa'])

def cargar_evaluaciones():
    """Carga las evaluaciones desde el archivo CSV actualizado"""
    try:
        df = pd.read_csv("evaluaciones.csv")
        # Eliminar columna si existe
        if 'articulo_referencia' in df.columns:
            df = df.drop(columns=['articulo_referencia'])
        return df
    except:
        return pd.DataFrame(columns=['fecha', 'nombre_alumno', 'calificacion', 'evaluacion_completa'])

def extraer_nombre_contenido(texto):
    """Extrae el nombre del alumno del contenido del PDF usando IA"""
    try:
        prompt_template = """
        Analiza el siguiente texto y extrae el nombre completo del alumno/estudiante/autor.
        Si encuentras múltiples nombres, selecciona el que más probablemente sea el autor del documento.
        Si no encuentras ningún nombre, responde exactamente con: "NO_ENCONTRADO"

        Reglas:
        1. El nombre debe contener al menos un nombre y un apellido
        2. Ignora nombres que sean claramente referencias bibliográficas
        3. Presta especial atención a frases como "presentado por", "autor:", "alumno:", etc.
        4. Devuelve solo el nombre, sin títulos (Dr., Ing., etc.) ni texto adicional
        5. Asegúrate de preservar tildes y caracteres especiales del nombre

        Texto a analizar:
        {texto}

        Nombre encontrado:"""

        llm = ChatOpenAI(temperature=0.1)
        prompt = PromptTemplate(template=prompt_template, input_variables=["texto"])
        chain = prompt | llm
        
        # Obtener solo las primeras 1000 palabras para el análisis
        texto_corto = ' '.join(texto.split()[:1000])
        resultado = chain.invoke({"texto": texto_corto})
        nombre = resultado.content.strip() if hasattr(resultado, 'content') else str(resultado).strip()
        
        # Validar que el nombre no esté vacío y no sea "NO_ENCONTRADO"
        if nombre and nombre != "NO_ENCONTRADO" and len(nombre.split()) >= 2:
            return nombre
        return None
    except Exception as e:
        st.warning(f"Error al extraer nombre usando IA: {str(e)}")
        return None

def procesar_pdf(archivo_pdf):
    """Procesa el PDF y extrae tanto el texto como el nombre del alumno"""
    try:
        # Verificar si es bytes o archivo
        contenido_pdf = archivo_pdf.getvalue() if hasattr(archivo_pdf, 'getvalue') else archivo_pdf
        
        # Verificar tipo MIME real
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(contenido_pdf)
        if file_type != 'application/pdf':
            raise ValueError("El archivo no es un PDF válido")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contenido_pdf)
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        paginas = loader.load()
        
        # Limpiar archivo temporal
        os.unlink(tmp_file_path)
        
        if not paginas:
            raise ValueError("El PDF está vacío o no se pudo leer")
            
        # Unir todas las páginas en un solo texto
        texto_completo = "\n".join([pagina.page_content for pagina in paginas])
        
        # Extraer nombre usando IA
        nombre_alumno = extraer_nombre_contenido(texto_completo)
            
        # Si no hay nombre, usar un nombre genérico
        if not nombre_alumno:
            nombre_alumno = "Estudiante " + datetime.now().strftime("%Y%m%d_%H%M%S")
            st.warning(f"No se pudo identificar el nombre del alumno. Se usará un identificador temporal: {nombre_alumno}")
            
        return texto_completo, nombre_alumno
        
    except Exception as e:
        # Asegurar que siempre devolvemos un nombre, incluso en caso de error
        nombre_temporal = "Estudiante " + datetime.now().strftime("%Y%m%d_%H%M%S")
        raise Exception(f"Error al procesar el PDF: {str(e)}. Se usará el identificador: {nombre_temporal}")

# Definir el estado del sistema
class AgentState(TypedDict):
    article_urls: List[str]
    processed_articles: Annotated[List[dict], operator.add]
    essay_content: str
    essay_data: dict
    evaluation_result: dict

# Agente de Carga de Artículos
class DocumentLoaderAgent:
    def process_article(self, state: AgentState) -> AgentState:
        processed = []
        for url in state['article_urls']:
            vectorstore, vector_id = procesar_articulo(url)
            processed.append({
                'id': vector_id,
                'url': url,
                'vectorstore': vectorstore
            })
        return {**state, 'processed_articles': processed}

# Agente de Análisis de Tareas
class TaskAnalyzerAgent:
    def process_submission(self, state: AgentState) -> AgentState:
        # Usar los datos pre-procesados si están disponibles
        if 'essay_data' in state and state['essay_data']:
            return state
            
        # Si no hay datos pre-procesados, procesar el PDF
        contenido_pdf = state['essay_content']
        if not isinstance(contenido_pdf, bytes):
            raise ValueError("El contenido del PDF debe ser bytes")
            
        texto_completo, nombre_alumno = procesar_pdf(contenido_pdf)
        return {**state, 'essay_data': {
            'content': texto_completo,
            'student_name': nombre_alumno
        }}

# Agente de Evaluación
class EvaluationAgent:
    def evaluate(self, state: AgentState) -> AgentState:
        # Obtener datos de los estados anteriores
        articles = state['processed_articles']
        essay_data = state['essay_data']
        
        # Combinar documentos de referencia
        docs_combinados = []
        for art in articles:
            vs = art['vectorstore']
            if hasattr(vs, 'docstore'):
                docs_combinados.extend([
                    doc for doc in vs.docstore._dict.values()
                    if hasattr(doc, 'page_content') and doc.page_content.strip()
                ])
        
        # Realizar evaluación
        resultado, calificacion = evaluar_ensayo(
            essay_data['content'],
            [art['vectorstore'] for art in articles],
            essay_data['student_name']
        )
        
        return {**state, 'evaluation_result': {
            'reporte': resultado,
            'calificacion': calificacion
        }}

def setup_workflow():
    workflow = StateGraph(AgentState)
    
    loader = DocumentLoaderAgent()
    analyzer = TaskAnalyzerAgent()
    evaluator = EvaluationAgent()
    
    workflow.add_node("cargar_articulos", loader.process_article)
    workflow.add_node("analizar_tarea", analyzer.process_submission)
    workflow.add_node("evaluar_ensayo", evaluator.evaluate)
    
    workflow.add_edge("cargar_articulos", "analizar_tarea")
    workflow.add_edge("analizar_tarea", "evaluar_ensayo")
    workflow.add_edge("evaluar_ensayo", END)
    
    workflow.set_entry_point("cargar_articulos")
    
    # Generar y guardar el diagrama Mermaid
    mermaid_code = """graph TD
    start((__start__)) --> cargar_articulos
    cargar_articulos[DocumentLoaderAgent] --> analizar_tarea
    analizar_tarea[TaskAnalyzerAgent] --> evaluar_ensayo
    evaluar_ensayo[EvaluationAgent] --> end((__end__))

    style start fill:#f9f,stroke:#333,stroke-width:2px
    style end fill:#f9f,stroke:#333,stroke-width:2px
    style cargar_articulos fill:#bbf,stroke:#333,stroke-width:2px
    style analizar_tarea fill:#bbf,stroke:#333,stroke-width:2px
    style evaluar_ensayo fill:#bbf,stroke:#333,stroke-width:2px"""
    
    with open("workflow_diagram.mmd", "w") as f:
        f.write(mermaid_code)
    
    return workflow.compile()

# Inicializar el workflow después de definir todas las clases y funciones necesarias
evaluation_workflow = setup_workflow()

# Interfaz principal
tab1, tab2, tab3, tab4 = st.tabs(["Subir Artículo", "Evaluar Tarea", "Listado de Evaluaciones", "Flujo de Agentes"])

with tab1:
    st.header("Subir Artículo de Referencia")
    st.subheader("Subir Nuevo Artículo")
    url = st.text_input("Ingresa la URL del artículo:")
    if st.button("Procesar Artículo") and url:
        with st.spinner("Procesando artículo..."):
            try:
                vectorstore, vector_id = procesar_articulo(url)
                if vectorstore and vector_id:
                    # Actualizar la lista de artículos en session_state
                    metadata = {
                        'id': vector_id,
                        'url': url,
                        'titulo': extraer_titulo_articulo(url),
                        'fecha_procesado': datetime.now().isoformat()
                    }
                    # Evitar duplicados
                    if not any(art['id'] == vector_id for art in st.session_state.articulos_procesados):
                        st.session_state['articulos_procesados'].append(metadata)
                        st.success("¡Artículo procesado y guardado exitosamente!")
                    else:
                        st.info("Este artículo ya estaba procesado anteriormente")
            except Exception as e:
                st.error(f"Error al procesar el artículo: {str(e)}")

with tab2:
    st.header("Evaluación de Tarea")
    
    if 'vectorstore' not in st.session_state:
        st.warning("Selecciona artículos de referencia en el sidebar")
    else:
        archivo_pdf = st.file_uploader("Subir PDF de tarea:", type=['pdf'])
        if archivo_pdf:
            with st.spinner("Procesando evaluación..."):
                try:
                    # Procesar el PDF primero para obtener el nombre del estudiante
                    texto_completo, nombre_alumno = procesar_pdf(archivo_pdf)
                    
                    # Ejecutar workflow con los datos procesados
                    result = evaluation_workflow.invoke({
                        'article_urls': [art['url'] for art in st.session_state.articulos_procesados],
                        'essay_content': archivo_pdf.getvalue(),
                        'essay_data': {
                            'content': texto_completo,
                            'student_name': nombre_alumno
                        }
                    })
                    
                    if 'evaluation_result' in result:
                        # Mostrar resultados
                        st.subheader("Resultado de la Evaluación")
                        st.write(result['evaluation_result']['reporte'])
                        st.write(f"**Calificación:** {result['evaluation_result']['calificacion']}")
                        
                        if st.button("Guardar Evaluación"):
                            guardar_evaluacion(
                                nombre_alumno,
                                result['evaluation_result']['calificacion'],
                                result['evaluation_result']['reporte']
                            )
                            st.success("Evaluación guardada")
                    else:
                        st.error("No se pudo completar la evaluación. Por favor, intenta nuevamente.")
                        
                except Exception as e:
                    st.error(f"Error en el flujo de evaluación: {str(e)}")

with tab3:
    st.header("Listado de Evaluaciones")
    
    # Cargar evaluaciones
    df = cargar_evaluaciones()
    
    if not df.empty:
        # Crear lista de alumnos con formato: "Nombre - Fecha - Calificación"
        opciones = [
            f"{row['nombre_alumno']} | {row['fecha']} | Calificación: {row['calificacion']:.1f}"
            for _, row in df.iterrows()
        ]
        
        # Selector de alumnos
        seleccion = st.selectbox(
            "Selecciona una evaluación para ver detalles:",
            opciones,
            index=0,
            key="selector_evaluaciones"
        )
        
        # Obtener el índice seleccionado
        idx_seleccionado = opciones.index(seleccion)
        
        # Mostrar detalles de la evaluación seleccionada
        st.markdown("---")
        st.subheader(f"Detalles de la Evaluación - {df.iloc[idx_seleccionado]['nombre_alumno']}")
        st.markdown(f'<div class="detalle-evaluacion">{df.iloc[idx_seleccionado]["evaluacion_completa"]}</div>', unsafe_allow_html=True)
        
    else:
        st.info("No hay evaluaciones guardadas todavía.")

with tab4:
    st.header("Flujo de Agentes de IA")
    
    st.subheader("¿Qué son los Agentes de IA?")
    st.markdown("""
    Los Agentes de IA (Agentic AI) son componentes de software autónomos que pueden:
    - Percibir su entorno a través de datos
    - Tomar decisiones basadas en su conocimiento
    - Ejecutar acciones que afectan su entorno
    - Colaborar con otros agentes para lograr objetivos complejos
    
    En este sistema, utilizamos agentes especializados que trabajan juntos para procesar y evaluar ensayos académicos.
    """)
    
    st.subheader("LangGraph")
    st.markdown("""
    LangGraph es una biblioteca que permite crear flujos de trabajo con agentes de IA de manera estructurada y eficiente. Características principales:
    - Creación de grafos de ejecución dirigidos
    - Gestión de estado entre agentes
    - Coordinación de múltiples agentes
    - Control de flujo y manejo de errores
    
    En nuestra aplicación, LangGraph orquesta el proceso de evaluación a través de tres agentes especializados.
    """)
    
    st.subheader("Diagrama de Flujo del Sistema")
    
    # Crear diagrama usando Graphviz
    graph = """
    digraph {
        rankdir=TB;
        node [shape=box, style=filled, fillcolor=lightblue, fontname=Arial];
        start [shape=circle, fillcolor=pink, label="__start__"];
        end [shape=circle, fillcolor=pink, label="__end__"];
        
        cargar_articulos [label="DocumentLoaderAgent"];
        analizar_tarea [label="TaskAnalyzerAgent"];
        evaluar_ensayo [label="EvaluationAgent"];
        
        start -> cargar_articulos;
        cargar_articulos -> analizar_tarea;
        analizar_tarea -> evaluar_ensayo;
        evaluar_ensayo -> end;
    }
    """
    
    st.graphviz_chart(graph)