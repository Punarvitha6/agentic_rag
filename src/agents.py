from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
from vectordb import DocumentSearchTool

class AWSRAGCrew:
    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
        self.search_tool = DocumentSearchTool()

    def run(self, user_query: str):
        # 1. Planner Agent (Requirement 2.1)
        planner = Agent(
            role='Architectural Planner',
            goal='Analyze query and determine retrieval strategy from document sections.',
            backstory='AWS Cloud Architect specialized in RAG strategies.',
            llm=self.llm,
            verbose=True
        )

        # 2. Retrieval Agent (Requirement 2.2)
        retriever_agent = Agent(
            role='Technical Researcher',
            goal='Extract precise passages and scores using the search tool.',
            backstory='Expert researcher who provides evidence-based context.',
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True
        )

        # 3. Answering Agent (Requirement 2.3)
        synthesizer = Agent(
            role='Synthesis specialist',
            goal='Generate a cited response grounded strictly in context.',
            backstory='Technical writer ensuring zero hallucinations.',
            llm=self.llm,
            verbose=True
        )

        # Tasks
        t1 = Task(description=f"Analyze: {user_query}. Decide which sections to search.", agent=planner, expected_output="Search plan.")
        t2 = Task(description=f"Retrieve facts for: {user_query}.", agent=retriever_agent, expected_output="Cited context snippets.")
        t3 = Task(
            description=(
                "Synthesize answer. Cite Page Numbers. "
                "If info is missing, say: 'This information is not available in the provided AWS RAG guide'."
            ),
            agent=synthesizer,
            context=[t1, t2],
            expected_output="Final Markdown response with citations."
        )

        crew = Crew(agents=[planner, retriever_agent, synthesizer], tasks=[t1, t2, t3], process=Process.sequential)
        return crew.kickoff()