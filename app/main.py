from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crews.pre_assessment_crew import PreAssessCrew
from crews.studey_materils_crew import StudyMeterial
from crews.ai_tutor_agents_crew import TutorCrew
from crews.learning_path_planner import PathPlnner
load_dotenv()
    
if __name__ == "__main__":
    pre_assessment = PreAssessCrew()
    study_materials = StudyMeterial()
    tutor = TutorCrew()
    path = PathPlnner()
    # pre_assessment.crew().kickoff(inputs={'topic':'Prompt Engineering'})
    # study_materials.crew().kickoff(
    #     inputs={
    #         'topic':'Prompt Engineering',
    #         'USER_LEVEL':'beginner',
    #         'competency': 'Write structured prompts, Optimize outputs iteratively, Control tone & format, Chain prompts logically, Handle edge cases, Reduce hallucinations, Apply task decomposition, Validate outputs, Use system instructions, Build reusable prompt templates'
    #         }
    #     )
    # tutor.crew().kickoff(
    #     inputs={
    #         'topic':'Prompt Engineering',
    #         'USER_LEVEL':'beginner',
    #         'competency': 'Write structured prompts, Optimize outputs iteratively, Control tone & format, Chain prompts logically, Handle edge cases, Reduce hallucinations, Apply task decomposition, Validate outputs, Use system instructions, Build reusable prompt templates'
    #         }
    #     )
    path.crew().kickoff(inputs={
            'topic':'Prompt Engineering',
            'USER_LEVEL':'beginner',
            'competency': 'Write structured prompts, Optimize outputs iteratively, Control tone & format, Chain prompts logically, Handle edge cases, Reduce hallucinations, Apply task decomposition, Validate outputs, Use system instructions, Build reusable prompt templates'
            })