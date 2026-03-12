from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class PathPlnner():
    '''Adaptive learning path planner for 22-step arcs'''

    agents_config = '../config/agents.yaml'
    tasks_config = '../config/tasks.yaml'

    @agent
    def path_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['learning_path_planner_agent'],
            llm=LLM(model='gpt-4o-mini', temperature=0.5),
            verbose=True,
        )

    @task
    def path_finder(self) -> Task:
        return Task(
            config=self.tasks_config['learning_path_planner_task'],
            agent=self.path_agent(),
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.path_agent()],
            tasks=[self.path_finder()],
            process=Process.sequential,
            verbose=True,
        )