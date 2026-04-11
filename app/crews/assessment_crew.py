from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, crew, agent, task

@CrewBase
class AssessmentCrew():
    '''Rubric-based assessment evaluator powered by Claude Sonnet'''

    agents_config = "../config/agents.yaml"
    tasks_config = '../config/tasks.yaml'

    @agent
    def evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['assessment_evaluator_agent'],
            llm=LLM(model='claude-sonnet-4-20250514', temperature=0.2),
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
