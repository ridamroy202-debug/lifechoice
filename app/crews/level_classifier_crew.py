from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class LevelClassifierCrew():
    '''Classifies learner level from pre-assessment responses'''
    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    @agent
    def classifier(self) -> Agent:
        return Agent(
            config=self.agents_config['level_classifier_agent'],
            llm=LLM(model='gpt-4o-mini'),
            verbose=False
        )

    @task
    def classify(self) -> Task:
        return Task(
            config=self.tasks_config['level_classifing_task'],
            agent=self.classifier()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.classifier()],
            tasks=[self.classify()],
            process=Process.sequential,
            verbose=False
        )
