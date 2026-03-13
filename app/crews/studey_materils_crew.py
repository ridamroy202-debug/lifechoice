from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class StudyMeterial():
    '''Professional study material generator for micro-credentials'''

    agents_config = "../config/agents.yaml"
    tasks_config = "../config/tasks.yaml"

    @agent
    def materials(self) -> Agent:
        return Agent(
            config=self.agents_config['senario_content_generator_agent'],
            llm=LLM(model='gpt-4o', temperature=0.6, max_tokens=8192),
            verbose=True,
        )

    @task
    def generate(self) -> Task:
        return Task(
            config=self.tasks_config['senario_generation_task'],
            agent=self.materials(),
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.materials()],
            tasks=[self.generate()],
            process=Process.sequential,
            verbose=True,
        )
