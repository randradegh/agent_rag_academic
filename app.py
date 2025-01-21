import streamlit as st
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

# Cargar variables de entorno
load_dotenv()

# Configurar OpenAI API Key desde .env
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.environ.get("OPENAI_API_KEY"):
    st.error("No se encontró la API Key de OpenAI en el archivo .env")
    st.stop()

# Configuración de la página
st.set_page_config(page_title="Evaluador de Ensayos", layout="wide")
st.title("Sistema de Evaluación de Ensayos")

# Instrucciones del proceso
st.markdown("""
### Instrucciones del Proceso
1. **Paso 1:** En la pestaña 'Subir Artículo', ingresa la URL del artículo de referencia y procésalo.
2. **Paso 2:** En la pestaña 'Evaluar Tarea', sube el archivo PDF con la tarea del estudiante.
3. **Paso 3:** El sistema evaluará automáticamente la tarea y proporcionará una calificación detallada.

**Nota:** Asegúrate de procesar primero el artículo de referencia antes de subir la tarea.
""")

# Configurar directorio para almacenar vectorstores
VECTOR_STORE_DIR = "vectorstores"
if not os.path.exists(VECTOR_STORE_DIR):
    os.makedirs(VECTOR_STORE_DIR)

# Configurar directorio de datos NLTK
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

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
        response = requests.head(url)
        return response.status_code == 200
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
    vectorstore.save_local(vector_path, allow_dangerous_deserialization=True)
    
    # Guardar metadata
    with open(os.path.join(VECTOR_STORE_DIR, f"{vector_id}_meta.txt"), "w") as f:
        f.write(url)
    
    return vector_id

def cargar_vectorstore(vector_id):
    """Carga un vectorstore desde disco"""
    vector_path = os.path.join(VECTOR_STORE_DIR, vector_id)
    if os.path.exists(vector_path):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
    return None

def listar_articulos_guardados():
    """Lista todos los artículos guardados"""
    articulos = []
    for archivo in os.listdir(VECTOR_STORE_DIR):
        if archivo.endswith("_meta.txt"):
            with open(os.path.join(VECTOR_STORE_DIR, archivo), "r") as f:
                url = f.read().strip()
            vector_id = archivo.replace("_meta.txt", "")
            articulos.append({"id": vector_id, "url": url})
    return articulos

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

def evaluar_ensayo(ensayo, vectorstore, nombre_alumno):
    """Evalúa el ensayo en dos pasos: primero el análisis y luego la calificación"""
    try:
        # Primer prompt para el análisis cualitativo
        prompt_analisis = """
        Eres un profesor experto evaluando ensayos académicos. Analiza el siguiente ensayo del estudiante {nombre_alumno}, 
        comparándolo con el artículo de referencia.
        
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

def guardar_evaluacion(nombre_alumno, calificacion, resultado, url_articulo):
    """Guarda la evaluación en un archivo CSV"""
    try:
        # Asegurar que la calificación sea un número válido
        if not isinstance(calificacion, (int, float)) or pd.isna(calificacion):
            calificacion = 7.0
        
        # Convertir explícitamente a float y redondear
        calificacion = round(float(calificacion), 1)
        
        evaluacion = {
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'nombre_alumno': str(nombre_alumno),
            'calificacion': calificacion,
            'articulo_referencia': str(url_articulo),
            'evaluacion_completa': str(resultado)
        }
        
        # Cargar evaluaciones existentes o crear nuevo DataFrame
        csv_path = "evaluaciones.csv"
        try:
            df = pd.read_csv(csv_path)
        except:
            df = pd.DataFrame(columns=['fecha', 'nombre_alumno', 'calificacion', 'articulo_referencia', 'evaluacion_completa'])
        
        # Crear un nuevo DataFrame con los tipos de datos correctos
        nueva_evaluacion = pd.DataFrame([evaluacion])
        
        # Asegurar que la columna de calificación sea numérica
        nueva_evaluacion['calificacion'] = pd.to_numeric(nueva_evaluacion['calificacion'], errors='coerce').fillna(7.0)
        
        # Concatenar los DataFrames
        df = pd.concat([df, nueva_evaluacion], ignore_index=True)
        
        # Asegurar que la columna de calificación sea numérica en el DataFrame final
        df['calificacion'] = pd.to_numeric(df['calificacion'], errors='coerce').fillna(7.0)
        
        df.to_csv(csv_path, index=False)
        return df
        
    except Exception as e:
        return pd.DataFrame(columns=['fecha', 'nombre_alumno', 'calificacion', 'articulo_referencia', 'evaluacion_completa'])

def cargar_evaluaciones():
    """Carga las evaluaciones desde el archivo CSV"""
    try:
        return pd.read_csv("evaluaciones.csv")
    except:
        return pd.DataFrame(columns=['fecha', 'nombre_alumno', 'calificacion', 'articulo_referencia', 'evaluacion_completa'])

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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(archivo_pdf.getvalue())
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

# Interfaz principal
tab1, tab2 = st.tabs(["Subir Artículo", "Evaluar Tarea"])

with tab1:
    st.header("Subir Artículo de Referencia")
    
    # Mostrar artículos guardados
    st.subheader("Artículos Guardados")
    articulos = listar_articulos_guardados()
    if articulos:
        for articulo in articulos:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {articulo['url']}")
            with col2:
                if st.button("Usar este artículo", key=articulo['id']):
                    vectorstore = cargar_vectorstore(articulo['id'])
                    if vectorstore:
                        st.session_state['vectorstore'] = vectorstore
                        st.success("¡Artículo cargado exitosamente!")
    else:
        st.info("No hay artículos guardados")
    
    st.divider()
    st.subheader("Subir Nuevo Artículo")
    url = st.text_input("Ingresa la URL del artículo:")
    if st.button("Procesar Artículo") and url:
        with st.spinner("Procesando artículo..."):
            try:
                vectorstore, vector_id = procesar_articulo(url)
                if vectorstore and vector_id:
                    st.session_state['vectorstore'] = vectorstore
                    st.success("¡Artículo procesado y guardado exitosamente!")
            except Exception as e:
                st.error(f"Error al procesar el artículo: {str(e)}")

with tab2:
    st.header("Evaluación de Tarea")
    
    # Mostrar tabla de evaluaciones
    st.subheader("Evaluaciones Realizadas")
    df_evaluaciones = cargar_evaluaciones()
    
    if not df_evaluaciones.empty:
        # Mostrar solo las columnas relevantes en la tabla
        df_display = df_evaluaciones[['fecha', 'nombre_alumno', 'calificacion', 'articulo_referencia']]
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No hay evaluaciones registradas")
    
    st.divider()
    st.subheader("Nueva Evaluación")
    
    # Validar que haya un artículo de referencia cargado
    if 'vectorstore' not in st.session_state:
        st.warning("Por favor, primero procesa un artículo de referencia en la pestaña anterior.")
        st.stop()
    
    # Formulario de evaluación
    archivo_pdf = st.file_uploader("Sube la tarea en formato PDF", type=['pdf'])
    
    if archivo_pdf:
        try:
            with st.spinner("Procesando tarea..."):
                # Procesar el PDF y obtener texto y nombre
                texto_tarea, nombre_alumno = procesar_pdf(archivo_pdf)
                
                if not texto_tarea or not nombre_alumno:
                    st.error("No se pudo extraer el contenido del PDF o el nombre del alumno.")
                    st.stop()
                
                # Mostrar el nombre detectado
                st.info(f"Nombre del alumno detectado: {nombre_alumno}")
                
                # Evaluar el ensayo
                resultado, calificacion = evaluar_ensayo(texto_tarea, st.session_state['vectorstore'], nombre_alumno)
                
                if not isinstance(calificacion, (int, float)):
                    st.error("Error en la calificación. Se asignará una calificación por defecto.")
                    calificacion = 7.0
                
                # Mostrar resultado
                st.write(f"### Evaluación de: {nombre_alumno}")
                st.write(resultado)
                
                # Obtener URL del artículo
                url_articulo = ""
                for articulo in listar_articulos_guardados():
                    if cargar_vectorstore(articulo['id']) == st.session_state['vectorstore']:
                        url_articulo = articulo['url']
                        break
                
                # Guardar evaluación
                df_actualizado = guardar_evaluacion(nombre_alumno, calificacion, resultado, url_articulo)
                st.success("Evaluación guardada exitosamente")
                st.stop()  # Detener la ejecución después del mensaje de éxito
                
        except Exception as e:
            st.error(f"Error al procesar la tarea: {str(e)}")
            st.stop()