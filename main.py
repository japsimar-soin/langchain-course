from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    information = """
    Gerald Jinx Mouse, known simply as Jerry, is an American character and one of the two titular characters in Metro-Goldwyn-Mayer's series of Tom and Jerry theatrical animated short films and other animated media, usually acting as the protagonist opposite his rival Tom Cat. Created by William Hanna and Joseph Barbera, Jerry is an anthropomorphic (but usually silent) brown house mouse, who first appeared in the 1940 MGM animated short Puss Gets the Boot.
    """

    summary_template = """given the information {information} about a character I want you to create: 
    1. A short summary 
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.1)
    # llm = ChatOllama(model="gemma3:270m", temperature=0.1)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
