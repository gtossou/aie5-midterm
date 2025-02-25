import chainlit as cl
import pymupdf4llm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from time import sleep
from tavily import TavilyClient
from dotenv import load_dotenv
import json
import os

# todo : from langchain_community.document_loaders import TavilyLoader
# load .env file to environment
load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
TAVILY_API_KEY = os.getenv("TAVILY_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def get_course_recommendations(jobs_context):
    search_query = f"Coursera specializations related to {jobs_context[0]} and {jobs_context[1]} . Only the links related to actual specializations"

    # Perform Tavily web search
    search_results = tavily_client.search(
        query=search_query,
        num_results=5,
        search_depth="advanced",
        include_answer="basic",
        include_domains=["COURSERA.ORG"],
    )

    # Extract course titles, links, and descriptions
    course_recommendations = []
    for result in search_results.get("results", []):
        if result.get("url").startswith(
            "https://www.coursera.org/specializations"
        ) or result.get("url").startswith(
            "https://www.coursera.org/professional-certificates"
        ):
            course_recommendations.append(
                {
                    "title": result.get("title"),
                    "url": result.get("url"),
                    "description": result.get("content"),
                }
            )

    return course_recommendations


@cl.on_message
async def main(message: cl.Message):
    await cl.Message(content=f"Received : {message.content}").send()


@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(
        content="Welcome to Tech Career Counselor. Let me setup the system"
    )
    await msg.send()

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_dim = 1536
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    RESUME_COLLECTION_NAME = "resume_embeddings"
    PERSONAS_COLLECTION_NAME = "personas"
    JOBS_COLLECTION_NAME = "job_definitions"

    client.recreate_collection(
        collection_name=RESUME_COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    print("✅ Collection setup complete!")
    sleep(4)

    msg.content = "The system is ready. Let me ask you a few questions to understand what you're looking for!"
    await msg.update()

    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a kind career counselor for tech roles who provides help for career advice. You are here to help the user find the best career path for them.
                Use the following context to answer accurately:\n\n{retrieved_context}""",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    files = None

    ####################################################################################################

    # 1️⃣ Mode de travail préféré ?
    work_mode = await cl.AskActionMessage(
        content="Quel est votre mode de travail préféré ?",
        actions=[
            cl.Action(
                name="remote",
                payload={"value": "remote"},
                label="À distance (full remote)",
            ),
            cl.Action(name="hybrid", payload={"value": "hybrid"}, label="Hybride"),
            cl.Action(
                name="onsite", payload={"value": "onsite"}, label="Sur site uniquement"
            ),
        ],
    ).send()

    if work_mode:
        work_mode_value = work_mode.get("payload", {}).get("value")

    # 2️⃣ Préférez-vous travailler seul ou en équipe ?
    teamwork_preference = await cl.AskActionMessage(
        content="Préférez-vous travailler seul ou en équipe ?",
        actions=[
            cl.Action(name="alone", payload={"value": "alone"}, label="Plutôt seul"),
            cl.Action(name="team", payload={"value": "team"}, label="Plutôt en équipe"),
            cl.Action(name="both", payload={"value": "both"}, label="Un peu des deux"),
        ],
    ).send()

    if teamwork_preference:
        teamwork_value = teamwork_preference.get("payload", {}).get("value")

    # 3️⃣ Quel niveau d’autonomie souhaitez-vous ?
    autonomy_level = await cl.AskActionMessage(
        content="Quel niveau d’autonomie souhaitez-vous ?",
        actions=[
            cl.Action(
                name="independent",
                payload={"value": "independent"},
                label="Indépendance totale",
            ),
            cl.Action(
                name="managed",
                payload={"value": "managed"},
                label="Encadré par un manager ou une équipe",
            ),
            cl.Action(
                name="balanced",
                payload={"value": "balanced"},
                label="Équilibre entre autonomie et encadrement",
            ),
        ],
    ).send()

    if autonomy_level:
        autonomy_value = autonomy_level.get("payload", {}).get("value")

    # 4️⃣ Quel niveau de créativité souhaitez-vous ?
    creativity_level = await cl.AskActionMessage(
        content="Quel niveau de créativité souhaitez-vous dans votre travail ?",
        actions=[
            cl.Action(name="high", payload={"value": "high"}, label="Très créatif"),
            cl.Action(
                name="medium",
                payload={"value": "medium"},
                label="Technique mais avec un peu de créativité",
            ),
            cl.Action(
                name="low",
                payload={"value": "low"},
                label="Pas forcément créatif, mais logique et analytique",
            ),
        ],
    ).send()

    if creativity_level:
        creativity_value = creativity_level.get("payload", {}).get("value")

    # 5️⃣ Quel type d’environnement de travail préférez-vous ?
    work_environment = await cl.AskActionMessage(
        content="Quel type d’environnement de travail préférez-vous ?",
        actions=[
            cl.Action(name="startup", payload={"value": "startup"}, label="Start-up"),
            cl.Action(
                name="corporate",
                payload={"value": "corporate"},
                label="Grande entreprise",
            ),
            cl.Action(
                name="freelance",
                payload={"value": "freelance"},
                label="Freelance / autoentrepreneur",
            ),
        ],
    ).send()

    if work_environment:
        work_environment_value = work_environment.get("payload", {}).get("value")

    # 6️⃣ Préférez-vous coder ou éviter le code ?
    coding_preference = await cl.AskActionMessage(
        content="Préférez-vous coder ou éviter le code ?",
        actions=[
            cl.Action(
                name="love_coding",
                payload={"value": "love_coding"},
                label="J’aime coder / je veux apprendre",
            ),
            cl.Action(
                name="some_code",
                payload={"value": "some_code"},
                label="Je veux un métier avec peu de code",
            ),
            cl.Action(
                name="no_code",
                payload={"value": "no_code"},
                label="Je préfère un métier sans code",
            ),
        ],
    ).send()

    if coding_preference:
        coding_value = coding_preference.get("payload", {}).get("value")

    # 7️⃣ Quel type de problèmes aimez-vous résoudre ?
    problem_solving = await cl.AskActionMessage(
        content="Quel type de problèmes aimez-vous résoudre ?",
        actions=[
            cl.Action(
                name="technical",
                payload={"value": "technical"},
                label="Problèmes techniques",
            ),
            cl.Action(
                name="strategic",
                payload={"value": "strategic"},
                label="Problèmes stratégiques",
            ),
            cl.Action(
                name="data", payload={"value": "data"}, label="Problèmes de données"
            ),
            cl.Action(
                name="human",
                payload={"value": "human"},
                label="Problèmes humains et organisationnels",
            ),
        ],
    ).send()

    if problem_solving:
        problem_value = problem_solving.get("payload", {}).get("value")

    # 8️⃣ Quel équilibre entre routine et nouveauté recherchez-vous ?
    routine_novelty = await cl.AskActionMessage(
        content="Quel équilibre entre routine et nouveauté recherchez-vous ?",
        actions=[
            cl.Action(
                name="stable",
                payload={"value": "stable"},
                label="J’aime la stabilité et les tâches bien définies",
            ),
            cl.Action(
                name="dynamic",
                payload={"value": "dynamic"},
                label="J’aime la nouveauté et l’adaptation constante",
            ),
            cl.Action(
                name="balanced",
                payload={"value": "balanced"},
                label="J’aime l’équilibre entre les deux",
            ),
        ],
    ).send()

    if routine_novelty:
        routine_novelty_value = routine_novelty.get("payload", {}).get("value")

    # ✅ Store all collected responses
    user_answers = {
        "work_mode": work_mode_value,
        "teamwork_preference": teamwork_value,
        "autonomy_level": autonomy_value,
        "creativity_level": creativity_value,
        "work_environment": work_environment_value,
        "coding_preference": coding_value,
        "problem_solving": problem_value,
        "routine_novelty": routine_novelty_value,
    }

    cl.user_session.set("personality_test_results", user_answers)

    # 🎯 Final message
    await cl.Message(
        content="✅ Merci pour vos réponses ! Nous analysons vos préférences."
    ).send()

    # 🔵 Retrieve the stored Runnable from session storage
    runnable = cast(Runnable, cl.user_session.get("runnable"))

    ####################################################################################################
    ##Persona analysis
    ##################################################################################################
    persona_vector_store = QdrantVectorStore(
        client=client,
        collection_name=PERSONAS_COLLECTION_NAME,
        embedding=embedding_model,
    )
    persona_retriever = persona_vector_store.as_retriever(search_kwargs={"k": 2})
    persona_search_results = persona_retriever.invoke(
        f"regarding  {str(user_answers)}, tell the user about the  persona that suit him "
    )
    # TODO: USE SESSIONS
    persona_retrieved_context = "\n\n".join(
        [doc.page_content for doc in persona_search_results]
    )

    question = "From the context, explain the 2 personas that suits the user and why.Don't include the jobs related to the persona"
    answer_persona = cl.Message(content="")
    persona_generated_response = ""
    # Run the pipeline using the retrieved Runnable
    async for chunk in runnable.astream(
        {"question": question, "retrieved_context": persona_retrieved_context},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await answer_persona.stream_token(chunk)
        persona_generated_response += chunk

    await answer_persona.send()

    ####################################################################################################
    ##Resume analysis
    ##################################################################################################

    await cl.Message(
        content="Let's know more about your previous experiences! You surely have some transferable skills."
    ).send()

    #     # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf"],
            max_size_mb=4,
            # timeout=180,
        ).send()
    ####################################################################################################
    ##Resume analysis
    ##################################################################################################
    file = files[0]

    loader = PyMuPDFLoader(file.path)
    resume_embeddings = loader.load()

    resume_vector_db = QdrantVectorStore.from_documents(
        documents=resume_embeddings,
        embedding=embedding_model,
        collection_name=RESUME_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    resume_vector_store = QdrantVectorStore(
        client=client,
        collection_name=RESUME_COLLECTION_NAME,
        embedding=embedding_model,
    )
    # todo: USE LANGCHAIN RETRIEVER
    resume_retriever = resume_vector_store.as_retriever(search_kwargs={"k": 10})
    # TODO: USE SESSIONS
    resume_search_results = resume_retriever.invoke(
        "Make a summarization of the skills and experiences of this user"
    )

    resume_retrieved_context = "\n\n".join(
        [doc.page_content for doc in resume_search_results]
    )

    question = "From the context, what are the skills or experiences that can be used in a tech career?"
    answer_resume = cl.Message(content="")
    resume_generated_response = ""

    # Run the pipeline using the retrieved Runnable
    async for chunk in runnable.astream(
        {"question": question, "retrieved_context": resume_retrieved_context},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await answer_resume.stream_token(chunk)
        resume_generated_response += chunk

    await answer_resume.send()
    ####################################################################################################
    ##Global analysis
    ##################################################################################################
    jobs_vector_store = QdrantVectorStore(
        client=client,
        collection_name=JOBS_COLLECTION_NAME,
        embedding=embedding_model,
    )
    jobs_retriever = jobs_vector_store.as_retriever(search_kwargs={"k": 2})
    jobs_search_results = jobs_retriever.invoke(
        f"tell the user about the job that suit him based on his resume transferable skills and his persona from this context {str(resume_generated_response) + ' ' + str(persona_generated_response)}"
    )

    jobs = []
    for job in jobs_search_results:
        jobs.append(json.loads(job.page_content)["job_name"])

    jobs_retrieved_context = "\n\n".join(
        [doc.page_content for doc in jobs_search_results]
    )

    question = "Explain the 2 jobs that suits the user and why"
    answer_jobs = cl.Message(content="")
    jobs_generated_response = ""
    # Run the pipeline using the retrieved Runnable
    async for chunk in runnable.astream(
        {
            "question": question,
            "retrieved_context": jobs_retrieved_context
            + " "
            + str(resume_generated_response)
            + " "
            + str(persona_generated_response),
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await answer_jobs.stream_token(chunk)
        jobs_generated_response += chunk

    await answer_jobs.send()

    ####################################################################################################
    ##Course recommendation
    ##################################################################################################

    # Voudrais tu que je te recommande des formations liées à ces carrières ?
    will_recommend = await cl.AskActionMessage(
        content="Voudriez-vous que je vous recommande des formations liées à ces carrières ?",
        actions=[
            cl.Action(name="oui", payload={"value": "oui"}, label="Oui"),
            cl.Action(name="non", payload={"value": "non"}, label="Non"),
        ],
    ).send()

    if will_recommend:
        will_recommend_value = will_recommend.get("payload", {}).get("value")
        if will_recommend_value == "oui":
            course_recommendations = get_course_recommendations(jobs)
            if course_recommendations:
                await cl.Message(
                    content="Voici quelques spécialisations Coursera qui pourraient vous intéresser :"
                ).send()
                for course in course_recommendations:
                    await cl.Message(
                        content=f"🔗 [{course['title']}]({course['url']})\n\n{course['description']}"
                    ).send()
            else:
                await cl.Message(
                    content="Désolé, je n'ai pas trouvé de spécialisations Coursera correspondant à vos intérêts."
                ).send()
        else:
            await cl.Message(content="D'accord, merci pour votre temps!").send()
