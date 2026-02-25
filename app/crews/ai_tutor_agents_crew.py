from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, crew, agent, task

@CrewBase
class TutorCrew():
    '''Personalized Tutor Crew'''

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    @agent
    def tutor(self) -> Agent:
        return Agent(
            config = self.agents_config['personal_ai_tutor_agent'],
            verbose = True
        )
    @task
    def tutor_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_tutor_task'],
            agent=self.tutor(),
            verbose=True
        )
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.tutor()],
            tasks=[self.tutor_task()],
            process=Process.sequential,
            verbose=True
        )