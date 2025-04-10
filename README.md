# Document Knowledge Extractor with CrewAI

## Descripción

This project implements an advanced document knowledge extraction system using artificial intelligence agents.The system analyzes any text document, extracts key concepts, identifies semantic relationships, and generates structured documentation such as concept maps, definitions, and frequently asked questions. [cite: 18] Using the CrewAI framework and local language models through Ollama, this system allows converting unstructured documents into accessible and organized knowledge resources. 

## Features

* **Intelligent document segmentation:** Breaks complex documents into coherent and manageable units 
* **Entity and concept extraction:** Automatically identifies key terms, people, organizations, and main concepts 
* **Relationship analysis:** Discovers and maps semantic connections between concepts 
* **Definition creation:** Generates a clear and precise glossary of important terms 
* **Question and answer identification:** Extracts implicit question-answer pairs from the document 
* **Concept map generation:** Creates structured schemas showing knowledge hierarchy and relationships 
* **Coherence validation:** Checks that all extracted information is accurate and consistent 
* **Knowledge integration:** Synthesizes results into a structured reference document 

## Requirements

* Python 3.8+ 
* OPENAI API key
* Ollama (with llama3.2 model or similar if you do not have OPENAI API key or want to run the solution locally) 
* Internet connection (for downloading packages) 

## Installation

1.  Clone this repository:

    ```bash
    git clone [https://github.com/your-username/knowledge-extractor.git](https://github.com/your-username/knowledge-extractor.git)
    cd knowledge-extractor
    ```
    
2.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
    
3.  Make sure Ollama is installed and running with the necessary model:

    ```bash
    # Install Ollama following instructions at ollama.com
    ollama pull llama3.2:1b
    ```
    
4.  Configure environment variables :
   
    ```bash
    # Create a .env file with necessary variables
    # For example:
    echo "SERPER_API_KEY=your-api-key" > .env #Only if using web search tools
    ```
    

## Project Structure

```
├── config/
│   ├── agents.yaml         # Agent role and personality definitions
│   └── tasks.yaml          # Task definitions for each agent
├── document_knowledge_crew.py  # Agent crew implementation
├── main.py                 # Main script to run the system
├── input_document.txt      # Input document to process (example)
├── requirements.txt        # Project dependencies
└── .env                    # Environment variables 
```

## Usage

1.  Place your text document in the `input_document.txt` file or specify another path in `main.py`. 
2.  Run the main script:

    ```bash
    python main.py
    ```
    
3.  The system will process the document in several stages, showing progress in the console. 
4.  Results will be saved in a file named `knowledge_output.md`. 

## Customization

### Modify Agent Roles

Edit the `config/agents.yaml` file to change the personality, goals, and skills of the agents. 

### Adjust Tasks

Modify the `config/tasks.yaml` file to change the specific instructions for each task. 

### Change the LLM Model

To use a different model, modify the LLM configuration in `document_knowledge_crew.py`:

```python
11m = LLM(
    model="ollama/other-model:version",
    base_url="http://localhost:11434"
)
```


## Internal Workflow

The system implements a sequential workflow where:

1.  The Document Reader segments the document into coherent units 
2.  The Entity Extractor identifies concepts and key terms 
3.  The Relationship Analyzer discovers connections between concepts 
4.  The Concept Definer creates precise definitions 
5.  The Q&A Identifier extracts implicit questions and answers 
6.  The Schema Modeler creates structured representations 
7.  The FAQ Writer refines frequently asked questions 
8.  The Validator reviews coherence and accuracy 
9.  The Integrator synthesizes everything into a final document 

Each agent specializes in its task and receives context from previous steps. 

## Limitations

* Very extensive documents may require more resources 
* Results depend on the quality of the LLM model used 
* Processing may take time due to the sequential nature 

## Contributing

Contributions are welcome. Please send a pull request or open an issue to discuss proposed changes. 

## License

MIT 

## Acknowledgements

* This project uses CrewAI for agent orchestration 

```
