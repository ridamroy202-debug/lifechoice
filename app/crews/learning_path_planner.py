from crewai import Agent, Crew, Task,Process
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class PathPlnner():
    '''Ai path planner crew'''

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    @agent
    def path_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['learning_path_planner_agent'],
            verbose = True
        )
    @task
    def path_finder(self) -> Task:
        return Task(
            config=self.tasks_config['learning_path_planner_task'],
            agent=self.path_agent(),
            verbose=True
        )
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.path_agent()],
            tasks = [self.path_finder()],
            process=Process.sequential,
            verbose=True
        )