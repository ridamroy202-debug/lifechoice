from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from app.settings import settings


@CrewBase
class PreAssessCrew():
    '''Pre-Assessment Diagnostic Crew'''

    agents_config = "../config/agents.yaml"
    tasks_config = "../config/tasks.yaml"

    @agent
    def materials(self) -> Agent:
        return Agent(
            config=self.agents_config['diagonistic_agent'],
            llm=LLM(model=settings.anthropic_model, provider="anthropic", temperature=0.5),
            verbose=False,
        )

    @task
    def materials_maker(self) -> Task:
        return Task(
            config=self.tasks_config['diagonistic_task'],
            agent=self.materials(),
            verbose=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.materials()],
            tasks=[self.materials_maker()],
            process=Process.sequential,
            verbose=False,
        )

# Module-level singleton
_pre_assess_crew_instance: Crew | None = None

def get_pre_assess_crew() -> Crew:
    global _pre_assess_crew_instance
    if _pre_assess_crew_instance is None:
        _pre_assess_crew_instance = PreAssessCrew().crew()
    return _pre_assess_crew_instance
