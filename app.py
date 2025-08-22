import requests
import json
import gradio as gr
from functions import count_tokens_simple, initialize_weaviate_and_get_retriever_with_images, summarize_conversation, tokenize_with_ollama
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Configuración de memoria
MAX_TOKENS = 300  # Límite antes de hacer resumen
SUMMARY_SIZE = 500  # Tamaño aproximado del resumen

# Schema de funciones disponibles
FUNCTION_SCHEMA = {
    "suma": {
        "name": "suma",
        "description": "Suma dos números y devuelve el resultado",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "Primer número a sumar"
                },
                "b": {
                    "type": "number", 
                    "description": "Segundo número a sumar"
                }
            },
            "required": ["a", "b"]
        }
    }
}

def execute_function(function_name, parameters):
    """Ejecuta una función basada en el nombre y parámetros proporcionados"""
    if function_name == "suma":
        try:
            a = parameters.get("a", 0)
            b = parameters.get("b", 0)
            result = a + b
            return {
                "success": True,
                "result": result,
                "message": f"La suma de {a} + {b} = {result}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Error al ejecutar la función suma"
            }
    else:
        return {
            "success": False,
            "error": f"Función '{function_name}' no encontrada",
            "message": "La función solicitada no está disponible"
        }

def parse_function_call(response_text):
    """
    Intenta parsear una respuesta del modelo para detectar llamadas a funciones
    Busca patrones JSON que contengan function_call
    """
    try:
        # Buscar patrones JSON en la respuesta
        import re
        json_pattern = r'\{[^{}]*"function_call"[^{}]*\}'
        matches = re.findall(json_pattern, response_text)
        
        if matches:
            # Intentar parsear el primer match como JSON
            function_call_data = json.loads(matches[0])
            return function_call_data
        
        # También intentar parsear toda la respuesta como JSON
        try:
            full_json = json.loads(response_text.strip())
            if "function_call" in full_json:
                return full_json
        except:
            pass
            
        return None
    except Exception as e:
        print(f"Error parseando function call: {e}")
        return None

def manage_memory(llm, chat_history):
    """Gestiona la memoria dinámica con resumen automático"""
    # Contar tokens actuales
    current_tokens = count_tokens_simple(chat_history)
    print(f"Tokens actuales: {current_tokens}")  # Para debug
    
    # Si excede el límite, hacer resumen
    if current_tokens > MAX_TOKENS:
        print("Límite de tokens excedido. Resumiendo conversación...")
        summarized_history = summarize_conversation(llm, chat_history)
        return summarized_history
    
    return chat_history

def initialize_rag():
    retriever = initialize_weaviate_and_get_retriever_with_images(data_folder='./data',collection_name='DocumentosPDFOllama')
    llm = OllamaLLM(model="qwen3:4b")

    contextualize_q_system_prompt = """Reformula la pregunta para que se entienda sin necesidad del historial previo. No respondas, solo reformula si es necesario."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Crear el prompt del sistema con llaves escapadas
    system_prompt = """Eres un asistente que responde preguntas basándose en documentos PDF que pueden incluir texto e imágenes. 

    FUNCIONES DISPONIBLES:
    - suma(a, b): Suma dos números y devuelve el resultado
    - Parámetros: a (número), b (número)
    - Ejemplo de uso: para sumar números

    INSTRUCCIONES:
    1. Responde únicamente usando la información del contexto proporcionado para preguntas sobre documentos.
    2. Si la pregunta no puede responderse con la información disponible, di que no tienes esa información en los documentos.
    3. Si necesitas realizar cálculos matemáticos (sumas), puedes usar las funciones disponibles.
    4. Para usar una función, responde ÚNICAMENTE con un JSON en este formato:
    {{
        "function_call": {{
        "name": "nombre_funcion",
        "parameters": {{
            "parametro1": valor1,
            "parametro2": valor2
        }}
        }}
    }}
    5. Si no necesitas usar ninguna función, responde normalmente con texto.

    Contexto de documentos: {context}"""
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, stuff_documents_chain)

    return rag_chain, llm

def process_question(message, history):
    """
    Procesa una pregunta usando el sistema RAG y mantiene el historial con memoria dinámica
    Incluye soporte para function calling
    """
    global chat_history, llm
    
    try:
        # Gestionar memoria antes de procesar la pregunta
        print(f"Gestionando memoria...")
        chat_history = manage_memory(llm, chat_history)
        
        # Invocar el sistema RAG
        response = rag_chain.invoke({
            "input": message,
            "chat_history": chat_history
        })
        
        # Procesar la respuesta (eliminar etiquetas de pensamiento si existen)
        respuesta = response['answer']
        if '</think>' in respuesta:
            respuesta = respuesta.split('</think>')[-1].strip()
        
        print(f"Respuesta del modelo: {respuesta}")
        
        # Verificar si es una llamada a función
        function_call_data = parse_function_call(respuesta)
        
        if function_call_data and "function_call" in function_call_data:
            # Ejecutar la función
            func_call = function_call_data["function_call"]
            function_name = func_call.get("name")
            parameters = func_call.get("parameters", {})
            
            print(f"Ejecutando función: {function_name} con parámetros: {parameters}")
            
            # Ejecutar la función
            function_result = execute_function(function_name, parameters)
            
            if function_result["success"]:
                respuesta_final = function_result["message"]
                # Agregar información adicional si es útil
                respuesta_final += f"\n\n(Resultado calculado: {function_result['result']})"
            else:
                respuesta_final = f"Error en la ejecución: {function_result['message']}"
            
            # Actualizar el historial con la pregunta, la función ejecutada y el resultado
            chat_history.extend([
                HumanMessage(content=message),
                AIMessage(content=f"Función ejecutada: {function_name}({parameters}) = {function_result.get('result', 'error')}")
            ])
            
            return respuesta_final
        else:
            # Respuesta normal sin function calling
            chat_history.extend([
                HumanMessage(content=message), 
                AIMessage(content=respuesta)
            ])
            
            return respuesta
        
    except Exception as e:
        error_msg = f"Error al procesar la pregunta: {str(e)}"
        print(error_msg)
        return error_msg
    
def clear_chat():
    """Limpia el historial de chat"""
    global chat_history
    chat_history = []
    return None

def get_memory_stats():
    """Obtiene estadísticas de la memoria actual"""
    global chat_history
    tokens = count_tokens_simple(chat_history)
    messages = len(chat_history)
    return f"Tokens: {tokens}/{MAX_TOKENS} | Mensajes: {messages}"

# Crear la interfaz de Gradio
with gr.Blocks(title="Sistema RAG con Function Calling", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Sistema RAG con Function Calling")
    gr.Markdown("Haz preguntas sobre tus documentos y realiza cálculos matemáticos. El modelo puede usar funciones automáticamente.")
    
    # Mostrar funciones disponibles
    gr.Markdown("### Funciones disponibles:")
    gr.Markdown("- **suma(a, b)**: Suma dos números")
    gr.Markdown("*Ejemplos: '¿Cuánto es 15 + 27?', 'Suma 100 y 250', 'Calcula 45 + 33'*")
    
    # Indicador de memoria
    memory_info = gr.Textbox(
        label="Estado de la Memoria",
        value="Tokens: 0/300 | Mensajes: 0",
        interactive=False,
        max_lines=1
    )
    
    chatbot = gr.Chatbot(
        height=500,
        placeholder="Las respuestas aparecerán aquí...",
        label="Conversación"
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Escribe tu pregunta aquí... (documentos o cálculos)",
            label="Tu pregunta",
            scale=4
        )
        submit_btn = gr.Button("Enviar", variant="primary", scale=1)
    
    with gr.Row():
        clear_btn = gr.Button("Limpiar conversación", variant="secondary")
        refresh_memory_btn = gr.Button("Actualizar memoria", variant="secondary")
    
    # Configurar los eventos
    def respond(message, chat_history_gradio):
        if not message.strip():
            return chat_history_gradio, "", get_memory_stats()
        
        # Procesar la pregunta
        bot_response = process_question(message, chat_history_gradio)
        
        # Actualizar el historial de Gradio
        chat_history_gradio.append((message, bot_response))
        
        return chat_history_gradio, "", get_memory_stats()
    
    def clear_conversation():
        clear_chat()
        return [], "", "Tokens: 0/300 | Mensajes: 0"
    
    def refresh_memory_stats():
        return get_memory_stats()
    
    # Eventos de los botones
    submit_btn.click(
        respond, 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg, memory_info]
    )
    
    msg.submit(
        respond, 
        inputs=[msg, chatbot], 
        outputs=[chatbot, msg, memory_info]
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, msg, memory_info]
    )
    
    refresh_memory_btn.click(
        refresh_memory_stats,
        outputs=[memory_info]
    )

if __name__ == "__main__":
    
    rag_chain, llm = initialize_rag()  # Ahora retorna también el LLM

    chat_history = []
    demo.launch(
        server_name="0.0.0.0",  # Permite acceso desde otras IPs
        server_port=7860,       # Puerto por defecto de Gradio
        share=False,            # Cambiar a True si quieres un enlace público
        debug=True              # Habilita el modo debug
    )