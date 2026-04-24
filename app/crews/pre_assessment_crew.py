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
            llm=LLM(model=settings.openai_default_model, temperature=0.5),
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
