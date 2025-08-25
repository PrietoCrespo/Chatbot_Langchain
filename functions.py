import base64
import os
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.vectorstores import Weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain.schema import Document
from langchain_ollama.llms import OllamaLLM
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure
import requests
import fitz  
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

def load_new_file(file_path, collection_name):
    """
    Carga un nuevo documento PDF a una colección existente de Weaviate
    
    Args:
        file_path (str): Ruta al archivo PDF
        collection_name (str): Nombre de la colección existente
    """
    client = weaviate.connect_to_local()
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontro el archivo: {file_path}")
    
    # Verificar que la coleccion existe
    if not client.collections.exists(collection_name):
        raise ValueError(f"La coleccion '{collection_name}' no existe.")
    
    # Obtener la coleccion existente
    collection = client.collections.get(collection_name)
    
    # Carga pdfs a través de directorio
    loader = PyPDFDirectoryLoader(file_path)
    docs = loader.load()
    
    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=350,
        chunk_overlap=50
    )
    chunked_documents = text_splitter.split_documents(docs)
    
    # Obtener nombre archivo
    file_name = Path(file_path).name
    
    # Obtener el siguiente chunk_id disponible (para no duplicar IDs)
    existing_docs = collection.query.fetch_objects(
        limit=10000
    )
    max_chunk_id = -1
    for obj in existing_docs.objects:
        if 'chunk_id' in obj.properties:
            max_chunk_id = max(max_chunk_id, obj.properties['chunk_id'])
    
    starting_chunk_id = max_chunk_id + 1 #TODO repasar esto
    
    # Indexar documentos
    with collection.batch.dynamic() as batch:
        for i, doc in enumerate(chunked_documents):
            batch.add_object(
                properties={
                    "text": doc.page_content,
                    "chunk_id": starting_chunk_id + i,
                    "source": file_name,
                    "length": len(doc.page_content)
                }
            )
        
    return True

def get_image_description_with_llava(imagen_base64, prompt="Describe esta imagen en detalle"):
    """
    Obtiene descripción de imagen usando LLaVA a través de Ollama
    """
    try:
        payload = {
            "model": "llava:latest",
            "prompt": prompt,
            "images": [imagen_base64],
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error al obtener descripción de imagen: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error procesando imagen: {e}")
        return None

def extract_images_and_text_from_pdf(pdf_path):
    """
    Extrae tanto texto como imágenes de un PDF
    """
    doc = fitz.open(pdf_path)
    extracted_content = []
    
    data_for_json = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Extraer texto
        text = page.get_text()
        if text.strip():
            extracted_content.append({
                'type': 'text',
                'content': text,
                'page': page_num + 1,
                'source': pdf_path
            })
            data_for_json += str(" " + text)

        # Extraer imágenes
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:
                    img_data = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    print(f"Procesando imagen {img_index + 1} de la pag {page_num + 1}")
                    
                    description = get_image_description_with_llava(
                        img_base64,  
                        "Describe esta imagen en detalle, incluyendo texto si lo hay, objetos, personas, colores, y cualquier información relevante."
                    )
                    
                    if description:
                        print(f"Descripción obtenida recortada: {description[:100]}")
                        data_for_json += str(" " + description)
                        extracted_content.append({
                            'type': 'image',
                            'content': description,
                            'page': page_num + 1,
                            'source': pdf_path,
                            'image_index': img_index,
                            'image_data': img_base64
                        })
                
                pix = None  # Liberar memoria
                
            except Exception as e:
                print(f"Error procesando imagen {img_index} en página {page_num + 1}: {e}")
                continue

    if data_for_json:
        documento_procesado = process_document_into_json(content=data_for_json)
        if documento_procesado:
            print(f"Documento procesado: {documento_procesado}")

    doc.close()
    return extracted_content


def get_files_from_db(query, collection_name):
    client = weaviate.connect_to_local()
    collection = client.collections.get(collection_name)

    query_embedding = get_ollama_embedding(query)
        
    if query_embedding:
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=2,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        print(f"Response: {response}")
        for i, obj in enumerate(response.objects, 1):
            print(f"\nResultado {i}:")
            print(f"Distancia: {obj.metadata.distance:.4f}")
            print(f"Texto: {obj.properties['text'][:200]}...")
    client.close()

def process_document_into_json(content):
    """Procesa un documento y lo convierte en un formato json estructurado"""

    # Construir el texto de la conversación
    llm_procesamiento = OllamaLLM(model="qwen3:4b")
    # Prompt para resumir
    process_prompt = f"""Procesa todo el contenido tratando de devolver un 
        json estructurado con toda la información posible.
        Devuelvelo en un formato jon válido. 
        En caso de no poder, devuelve un string vacio ("").
        
        *Instrucciones estrictas:*
        1. No respondas a nada, simplemente devuelve JSON o "" 

        Contenido:
        {content}
        """
    
    try:
        print(f"Procesando documento y convirtiendolo a json...")
        document_processed = llm_procesamiento.invoke(process_prompt).split("</think>")[1]
        return document_processed
    except Exception as e:
        print(f"Error al procesar documento: {e}") 

def initialize_weaviate_and_get_retriever_with_images(data_folder, collection_name="DocumentosPDFOllama"):
    """
    Inicializa Weaviate, carga documentos incluyendo imágenes, los vectoriza con Ollama
    y devuelve un objeto Retriever de LangChain.
    """
    client = weaviate.connect_to_local()
    
    if client.collections.exists(collection_name): # Para asegurarme de que siempre la creo de nuevo
        client.collections.delete(collection_name)
        
   
    collection = client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="length", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="content_type", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="page_number", data_type=wvc.config.DataType.INT),
            wvc.config.Property(name="image_data", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="image_index", data_type=wvc.config.DataType.INT),
        ]
        
    )
    
    pdf_files = list(Path(data_folder).glob("*.pdf"))
    all_content = []
    
    for pdf_file in pdf_files:
        #print(f"Nombre pdf: {pdf_file.name}")
        content = extract_images_and_text_from_pdf(str(pdf_file))
        
        all_content.extend(content)
        

    # Crear documentos de LangChain
    text_documents = []
    for item in all_content: # Cada elemento es un pdf
        if item['type'] == 'text':
            doc = Document(
                page_content=item['content'],
                metadata={
                    'source': item['source'],
                    'page': item['page'],
                    'content_type': 'text'
                }
            )
            print(f"Documento creado a partir de --texto--")
            text_documents.append(doc)

    # Dividir texto en chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=350, chunk_overlap=50
    )
    chunked_text_documents = text_splitter.split_documents(text_documents)
    
    print(f"Documentos divididos en {len(chunked_text_documents)} chunks")
    
    
    chunk_id = 0
    
    with collection.batch.dynamic() as batch:
        # Insertar chunks de texto
        for doc in chunked_text_documents:
            embedding = get_ollama_embedding(doc.page_content)
            
            if embedding:
                file_name = Path(doc.metadata.get('source', 'unknown_source.pdf')).name
                batch.add_object(
                    properties={
                        "text": doc.page_content,
                        "chunk_id": chunk_id,
                        "source": file_name,
                        "length": len(doc.page_content),
                        "content_type": "text",
                        "page_number": doc.metadata.get('page', 0),
                        "image_data": "",  # Vacío para texto
                        "image_index": -1  # -1 para texto
                    },
                    vector=embedding
                )
                chunk_id += 1
        
        # Insertar descripciones de imágenes
        for item in all_content:
            if item['type'] == 'image':
                embedding = get_ollama_embedding(item['content'])
                
                if embedding:
                    file_name = Path(item['source']).name
                    batch.add_object(
                        properties={
                            "text": f"IMAGEN: {item['content']}",  # Descripcion de la imagen
                            "chunk_id": chunk_id,
                            "source": file_name,
                            "length": len(item['content']),
                            "content_type": "image",
                            "page_number": item['page'],
                            "image_data": item.get('image_data', ''),
                            "image_index": item.get('image_index', -1)
                        },
                        vector=embedding
                    )
                    chunk_id += 1
    
    total_items = len(chunked_text_documents) + len([x for x in all_content if x['type'] == 'image'])
    print(f"Se han insertado {total_items} (texto + imágenes) con embeddings de Ollama.") # Aunque si a alguno no le ha calculado los embeddings no lo
                                                                                          # cuenta realmente, habria que hacer suma en el for
    
    ollama_embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    
    weaviate_vectorstore = WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="text",
        embedding=ollama_embeddings_model,
    )
    
    # Crear retriever con más resultados para incluir tanto texto como imágenes
    retriever = weaviate_vectorstore.as_retriever(search_kwargs={"k": 5})
    print(f"Retriever de LangChain creado para la colección '{collection_name}' con soporte para imágenes")
    
    return retriever


def check_ollama_connection():
    """Verifica que Ollama esté ejecutándose"""
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            print("Modelos de Ollama disponibles:")
            for model in model_names:
                print(f"   - {model}")
            
            return len(model_names) > 0
        else:
            return False
            
    except requests.exceptions.RequestException:
        print("No se pudo conectar a Ollama")
        return False

def count_tokens_simple(messages):
    """Cuenta tokens de forma aproximada (4 caracteres = 1 token)"""
    total_chars = sum(len(msg.content) for msg in messages)
    return total_chars // 4

def summarize_conversation(llm, messages):
    """Resume la conversación manteniendo el contexto importante"""
    #TODO comprobar que esto lo haga bien
    # Construir el texto de la conversación
    conversation_text = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_text += f"Usuario: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_text += f"Asistente: {msg.content}\n"
    
    # Prompt para resumir
    summary_prompt = f"""Resume la siguiente conversación manteniendo:
        1. Los temas principales discutidos
        2. Información clave mencionada
        3. El contexto necesario para continuar la conversación
        4. Máximo {os.getenv('SUMMARY_SIZE')} tokens

        Conversación:
        {conversation_text}

        Resumen:"""
    
    try:
        summary = llm.invoke(summary_prompt).split("</think>")[1]
        return [SystemMessage(content=f"Resumen de conversación previa: {summary}")]
    except Exception as e:
        print(f"Error al resumir: {e}")
        # Si falla el resumen, mantener solo los últimos mensajes
        return messages[-4:]  # Últimos 2 intercambios
    
def get_ollama_embedding(text):
    """Obtener embedding de Ollama directamente"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text:latest",  # Usar el modelo que tienes disponible
                "prompt": text
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Error en Ollama: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error conectando a Ollama: {e}")
        return None

def tokenize_with_ollama(text, model_name="llama2:latest"):
    """Tokenizar usando la API de Ollama"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model_name,
        "prompt": text,
        "stream": False,
        "options": {
            "num_predict": 0,  # No generar texto, solo tokenizar
        }
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        if result["prompt_eval_count"]:
            return result["prompt_eval_count"]
        else:
            raise Exception(f"No ha calculado los tokens")
        # Ollama incluye información sobre tokens en la respuesta
    else:
        raise Exception(f"Error: {response.status_code}")