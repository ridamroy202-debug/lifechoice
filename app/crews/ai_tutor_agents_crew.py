from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, crew, agent, task

from app.settings import settings

@CrewBase
class TutorCrew():
    '''Personalized Tutor Crew — delivers production-grade teaching content'''

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    @agent
    def tutor(self) -> Agent:
        return Agent(
            config=self.agents_config['personal_ai_tutor_agent'],
            llm=LLM(model=settings.anthropic_model, provider="anthropic", temperature=0.45, max_tokens=6144),
            verbose=False,
            max_iter=3,
        )

    @task
    def tutor_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_tutor_task'],
            agent=self.tutor(),
            verbose=False,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.tutor()],
            tasks=[self.tutor_task()],
            process=Process.sequential,
            verbose=False,
        )
