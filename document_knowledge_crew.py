# document_knowledge_crew.py
from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import FileReadTool
from crewai import LLM
import os
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load YAML configurations
def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# # Configure LLM - Switch from OpenAI to Ollama
# llm = LLM(
#     model="ollama/llama3.2",  
#     base_url="http://localhost:11434"
# )


llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)


@CrewBase
class DocumentKnowledgeCrew:
    """
    A crew specialized in extracting structured knowledge from documents.
    This crew implements a complete workflow to analyze documents,
    extract key concepts, identify relationships, and generate useful documentation.
    """
    
    def __init__(self, document_path):
        """
        Initializes the crew with the document to process.
        
        Args:
            document_path: Path to the document to analyze
        """
        self.llm = llm
        self.document_path = document_path
        self.file_tool = FileReadTool()
        
        # Load YAML configurations
        self.agents_config = load_yaml('config/agents.yaml')
        self.tasks_config = load_yaml('config/tasks.yaml')
    
    # === Agent Definitions ===
    
    @agent
    def document_reader_agent(self) -> Agent:
        """Agent specialized in analyzing and segmenting document structure."""
        return Agent(
            role=self.agents_config['document_reader']['role'],
            goal=self.agents_config['document_reader']['goal'],
            backstory=self.agents_config['document_reader']['backstory'],
            llm=self.llm,
            tools=[self.file_tool],
            verbose=True
        )
    
    @agent
    def entity_extractor_agent(self) -> Agent:
        """Agent specialized in identifying entities and key concepts."""
        return Agent(
            role=self.agents_config['entity_extractor']['role'],
            goal=self.agents_config['entity_extractor']['goal'],
            backstory=self.agents_config['entity_extractor']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def relationship_analyzer_agent(self) -> Agent:
        """Agent specialized in identifying semantic relationships between concepts."""
        return Agent(
            role=self.agents_config['relationship_analyzer']['role'],
            goal=self.agents_config['relationship_analyzer']['goal'],
            backstory=self.agents_config['relationship_analyzer']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def concept_definer_agent(self) -> Agent:
        """Agent specialized in defining identified concepts."""
        return Agent(
            role=self.agents_config['concept_definer']['role'],
            goal=self.agents_config['concept_definer']['goal'],
            backstory=self.agents_config['concept_definer']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def qa_identifier_agent(self) -> Agent:
        """Agent specialized in identifying implicit questions and answers."""
        return Agent(
            role=self.agents_config['qa_identifier']['role'],
            goal=self.agents_config['qa_identifier']['goal'],
            backstory=self.agents_config['qa_identifier']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def schema_modeler_agent(self) -> Agent:
        """Agent specialized in creating concept maps/knowledge schemas."""
        return Agent(
            role=self.agents_config['schema_modeler']['role'],
            goal=self.agents_config['schema_modeler']['goal'],
            backstory=self.agents_config['schema_modeler']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def faq_writer_agent(self) -> Agent:
        """Agent specialized in writing FAQs based on document content."""
        return Agent(
            role=self.agents_config['faq_writer']['role'],
            goal=self.agents_config['faq_writer']['goal'],
            backstory=self.agents_config['faq_writer']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def validator_agent(self) -> Agent:
        """Agent specialized in validating and refining other agents' results."""
        return Agent(
            role=self.agents_config['validator']['role'],
            goal=self.agents_config['validator']['goal'],
            backstory=self.agents_config['validator']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    @agent
    def knowledge_integrator_agent(self) -> Agent:
        """Agent specialized in integrating all results into a cohesive knowledge document."""
        return Agent(
            role=self.agents_config['knowledge_integrator']['role'],
            goal=self.agents_config['knowledge_integrator']['goal'],
            backstory=self.agents_config['knowledge_integrator']['backstory'],
            llm=self.llm,
            verbose=True
        )
    
    # === Task Definitions ===
    
    @task
    def segment_document_task(self) -> Task:
        """Task for analyzing and segmenting the document."""
        description = self.tasks_config['segment_document']['description']
        expected_output = self.tasks_config['segment_document']['expected_output']
        
        # Add document path to description
        description = f"Read and analyze the document at path: {self.document_path}.\n\n" + description
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.document_reader_agent(),
            tools=[self.file_tool]
        )
    
    @task
    def extract_entities_task(self) -> Task:
        """Task for extracting entities and key concepts."""
        description = self.tasks_config['extract_entities']['description']
        expected_output = self.tasks_config['extract_entities']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.entity_extractor_agent(),
            context=[self.segment_document_task()]
        )
    
    @task
    def analyze_relationships_task(self) -> Task:
        """Task for analyzing relationships between concepts."""
        description = self.tasks_config['analyze_relationships']['description']
        expected_output = self.tasks_config['analyze_relationships']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.relationship_analyzer_agent(),
            context=[self.extract_entities_task()]
        )
    
    @task
    def define_concepts_task(self) -> Task:
        """Task for defining identified concepts."""
        description = self.tasks_config['define_concepts']['description']
        expected_output = self.tasks_config['define_concepts']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.concept_definer_agent(),
            context=[self.extract_entities_task()]
        )
    
    @task
    def identify_qa_pairs_task(self) -> Task:
        """Task for identifying questions and answers in the document."""
        description = self.tasks_config['identify_qa_pairs']['description']
        expected_output = self.tasks_config['identify_qa_pairs']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.qa_identifier_agent(),
            context=[self.segment_document_task()]
        )
    
    @task
    def create_schema_task(self) -> Task:
        """Task for creating a knowledge schema or concept map."""
        description = self.tasks_config['create_schema']['description']
        expected_output = self.tasks_config['create_schema']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.schema_modeler_agent(),
            context=[self.extract_entities_task(), self.analyze_relationships_task(), self.segment_document_task()]
        )
    
    @task
    def write_faqs_task(self) -> Task:
        """Task for writing FAQs based on document content."""
        description = self.tasks_config['write_faqs']['description']
        expected_output = self.tasks_config['write_faqs']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.faq_writer_agent(),
            context=[self.identify_qa_pairs_task()]
        )
    
    @task
    def validate_outputs_task(self) -> Task:
        """Task for validating all outputs."""
        description = self.tasks_config['validate_outputs']['description']
        expected_output = self.tasks_config['validate_outputs']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.validator_agent(),
            context=[
                self.segment_document_task(), 
                self.extract_entities_task(),
                self.analyze_relationships_task(), 
                self.define_concepts_task(),
                self.identify_qa_pairs_task(), 
                self.create_schema_task(),
                self.write_faqs_task()
            ]
        )
    
    @task
    def integrate_knowledge_task(self) -> Task:
        """Task for integrating all results into a final document."""
        description = self.tasks_config['integrate_knowledge']['description']
        expected_output = self.tasks_config['integrate_knowledge']['expected_output']
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.knowledge_integrator_agent(),
            context=[
                self.segment_document_task(), 
                self.extract_entities_task(),
                self.analyze_relationships_task(), 
                self.define_concepts_task(),
                self.identify_qa_pairs_task(), 
                self.create_schema_task(),
                self.write_faqs_task(),
                self.validate_outputs_task()
            ]
        )
    
    @crew
    def crew(self) -> Crew:
        """Defines the crew and its execution process."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )