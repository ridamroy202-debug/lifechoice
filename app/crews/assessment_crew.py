from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, crew, agent, task

from app.settings import settings

@CrewBase
class AssessmentCrew():
    '''Rubric-based assessment evaluator powered by Claude Sonnet'''

    agents_config = "../config/agents.yaml"
    tasks_config = '../config/tasks.yaml'

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['assessment_evaluator_agent'],
            llm=LLM(model=settings.anthropic_model, provider="anthropic", temperature=0.2),
            verbose=False,
        )

    @task
    def evaluate(self) -> Task:
        return Task(
            config=self.tasks_config['assessment_eval_task'],
            agent=self.evaluator(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.evaluator()],
            tasks=[self.evaluate()],
            process=Process.sequential,
            verbose=False,
        )

# Module-level singleton
_assessment_crew_instance: Crew | None = None

def get_assessment_crew() -> Crew:
    global _assessment_crew_instance
    if _assessment_crew_instance is None:
        _assessment_crew_instance = AssessmentCrew().crew()
    return _assessment_crew_instance
