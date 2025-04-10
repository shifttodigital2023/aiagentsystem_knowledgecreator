# main.py
from document_knowledge_crew import DocumentKnowledgeCrew
import os
from IPython.display import Markdown, display

def main():
    # Ruta al documento a procesar
    document_path = "input_document.txt"  # Ajusta según sea necesario
    
    # Verificar que existe
    if not os.path.exists(document_path):
        print(f"Error: El documento '{document_path}' no existe.")
        return
    
    print(f"Iniciando extracción de conocimiento del documento: {document_path}")
    
    # Crear el crew
    knowledge_crew = DocumentKnowledgeCrew(document_path)
    
    # Ejecutar el crew y obtener resultado
    response = knowledge_crew.crew().kickoff()
    
   # Guardar resultado
    output_path = "knowledge_output.md"
    with open(output_path, "w", encoding="utf-8") as f:
        if hasattr(response, 'raw'):
            # Si estás en un Jupyter Notebook y quieres mostrar el markdown:
            # from IPython.display import Markdown, display
            # display(Markdown(response.raw))
            
            # Guardar el contenido raw en el archivo
            f.write(response.raw)
            print(f"Extracción de conocimiento completada. Resultados guardados en: {output_path}")
        else:
            print("No se pudo extraer el texto del CrewOutput.")
            print("Tipo de respuesta:", type(response))
            print("Contenido de respuesta:", response)
            
            # Intentar convertir la respuesta a string como último recurso
            try:
                f.write(str(response))
                print(f"Se ha guardado la representación en string de la respuesta en: {output_path}")
            except Exception as e:
                print(f"Error al guardar la respuesta: {e}")
    
    print(f"Extracción de conocimiento completada. Resultados guardados en: {output_path}")

if __name__ == "__main__":
    main()