import os
import asyncio
import pytest
from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict
from langsmith import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

# Load environment variables
load_dotenv()

# =====================
# CONFIG & CONSTANTS
# =====================
DATASET_NAME = "Arts2Survive Blogs Q&A"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =====================
# RAG AGENT SETUP
# =====================
from src.agents.rag_agent import retriever

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=1)

@traceable()
def rag_bot(question: str) -> dict:
    """RAG bot that retrieves documents and generates answers"""
    # LangChain retriever will be automatically traced
    docs = retriever.invoke(question)
    docs_string = " \n".join(doc.page_content for doc in docs)

    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions. 
    Use the following source documents to answer the user's questions. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Documents:
    {docs_string}"""

    # langchain ChatModel will be automatically traced
    ai_msg = llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ])

    return {"answer": ai_msg.content, "documents": docs}

# =====================
# DATASET SETUP
# =====================
client = Client()

def get_or_create_dataset():
    """Create or load the test dataset"""
    examples = [
        {
            "inputs": {"question": "What are the main components covered in the LangChain features article?"},
            "outputs": {"answer": "The article discusses building an end‑to‑end pipeline including agents, MCP, LangGraph ToolNode integration, agent evaluations, and CI/CD deployment."}
        },
        {
            "inputs": {"question": "Why is evaluation important in LangChain workflows?"},
            "outputs": {"answer": "Evaluation is important because it measures performance, identifies gaps, and enables improvement over time using datasets, target functions, and metrics."}
        },
        {
            "inputs": {"question": "What went wrong during the F‑1 visa interview in Delhi with Pankaj?"},
            "outputs": {"answer": "Pankaj experienced multiple issues that started in the morning, leading to the rejection of the visa application."}
        },
        {
            "inputs": {"question": "What was the outcome of the second F‑1 visa attempt?"},
            "outputs": {"answer": "On the second attempt, the applicant passed the interview and obtained the F‑1 visa successfully."}
        },
        {
            "inputs": {"question": "How did Pankaj prepare differently for the second visa?"},
            "outputs": {"answer": "Pankaj reflected on the first rejection, prepared thoroughly, and confidently answered standard interview questions, which led to approval."}
        },
        {
            "inputs": {"question": "What criticism did Pankaja receive about their career choices?"},
            "outputs": {"answer": "Pankaj was told he was 'not loyal to any one field' and might lack productivity by switching roles frequently."}
        },
        {
            "inputs": {"question": "How did Pankaj react to the loyalty criticism?"},
            "outputs": {"answer": "Pankaj reflected deeply on those words and thought about how they are perceived by others during different phases of life."}
        }
    ]
    
    try:
        dataset = client.create_dataset(dataset_name=DATASET_NAME)
        client.create_examples(
            dataset_id=dataset.id,
            examples=examples
        )
        print("Dataset created and examples added...")
    except Exception:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print("Dataset already exists. loaded it...")
    return dataset

# =====================
# EVALUATION SCHEMAS
# =====================
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

# =====================
# EVALUATION PROMPTS
# =====================
correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# =====================
# EVALUATION FUNCTIONS
# =====================
def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """An evaluator for RAG answer accuracy"""
    grader_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        temperature=0
    ).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
    
    answers = f"""\
                    QUESTION: {inputs['question']}
                    GROUND TRUTH ANSWER: {reference_outputs['answer']}
                    STUDENT ANSWER: {outputs['answer']}"""

    # Run evaluator
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions}, 
        {"role": "user", "content": answers}
    ])
    return grade["correct"]

def relevance(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer helpfulness."""
    relevance_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        temperature=0
    ).with_structured_output(RelevanceGrade, method="json_schema", strict=True)
    
    answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]

def groundedness(inputs: dict, outputs: dict) -> bool:
    """A simple evaluator for RAG answer groundedness."""
    grounded_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        temperature=0
    ).with_structured_output(GroundedGrade, method="json_schema", strict=True)
    
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["grounded"]

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    """An evaluator for document relevance"""
    retrieval_relevance_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", 
        temperature=0
    ).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)
    
    doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
    answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"

    # Run evaluator
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions}, 
        {"role": "user", "content": answer}
    ])
    return grade["relevant"]

# =====================
# TARGET FUNCTION
# =====================
def target(inputs: dict) -> dict:
    """Target function for evaluation"""
    return rag_bot(inputs["question"])

# =====================
# TEST FUNCTIONS
# =====================
def test_rag_correctness():
    """Test RAG correctness evaluation"""
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness],
        experiment_prefix="rag-correctness-test"
    )
    
    # Get feedback scores
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=experiment_results.experiment_name)],
        feedback_key="correctness"
    )
    scores = [f.score for f in feedback]
    
    # Assert minimum correctness score
    assert len(scores) > 0, "No correctness scores found"
    avg_score = sum(scores) / len(scores)
    assert avg_score >= 0.7, f"Correctness score {avg_score:.2f} below 70% threshold"

def test_rag_relevance():
    """Test RAG relevance evaluation"""
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[relevance],
        experiment_prefix="rag-relevance-test"
    )
    
    # Get feedback scores
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=experiment_results.experiment_name)],
        feedback_key="relevance"
    )
    scores = [f.score for f in feedback]
    
    # Assert minimum relevance score
    assert len(scores) > 0, "No relevance scores found"
    avg_score = sum(scores) / len(scores)
    assert avg_score >= 0.8, f"Relevance score {avg_score:.2f} below 80% threshold"


def test_rag_groundedness():
    """Test RAG groundedness evaluation"""
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[groundedness],
        experiment_prefix="rag-groundedness-test"
    )
    
    # Get feedback scores
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=experiment_results.experiment_name)],
        feedback_key="groundedness"
    )
    scores = [f.score for f in feedback]
    
    # Assert minimum groundedness score
    assert len(scores) > 0, "No groundedness scores found"
    avg_score = sum(scores) / len(scores)
    assert avg_score >= 0.8, f"Groundedness score {avg_score:.2f} below 80% threshold"

def test_rag_retrieval_relevance():
    """Test RAG retrieval relevance evaluation"""
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[retrieval_relevance],
        experiment_prefix="rag-retrieval-relevance-test"
    )
    
    # Get feedback scores
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=experiment_results.experiment_name)],
        feedback_key="retrieval_relevance"
    )
    scores = [f.score for f in feedback]
    
    # Assert minimum retrieval relevance score
    assert len(scores) > 0, "No retrieval relevance scores found"
    avg_score = sum(scores) / len(scores)
    assert avg_score >= 0.7, f"Retrieval relevance score {avg_score:.2f} below 70% threshold"

def test_rag_all_dimensions():
    """Test RAG evaluation across all dimensions"""
    experiment_results = client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness, groundedness, relevance, retrieval_relevance],
        experiment_prefix="rag-comprehensive-test"
    )
    
    # Get feedback for all dimensions
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=experiment_results.experiment_name)]
    )
    
    # Group scores by dimension
    scores_by_dimension = {}
    for f in feedback:
        if f.key not in scores_by_dimension:
            scores_by_dimension[f.key] = []
        scores_by_dimension[f.key].append(f.score)
    
    # Assert minimum scores for each dimension
    thresholds = {
        "correctness": 0.7,
        "relevance": 0.8,
        "groundedness": 0.8,
        "retrieval_relevance": 0.7
    }
    
    for dimension, threshold in thresholds.items():
        if dimension in scores_by_dimension:
            scores = scores_by_dimension[dimension]
            avg_score = sum(scores) / len(scores)
            assert avg_score >= threshold, f"{dimension} score {avg_score:.2f} below {threshold*100}% threshold"
        else:
            pytest.fail(f"No scores found for {dimension}")

# =====================
# UTILITY FUNCTIONS
# =====================
def test_rag_bot_single_query():
    """Test RAG bot with a single query"""
    question = "What are the main components covered in the LangChain features article?"
    result = rag_bot(question)
    
    # Assert response structure
    assert "answer" in result, "Response missing 'answer' field"
    assert "documents" in result, "Response missing 'documents' field"
    assert isinstance(result["answer"], str), "Answer should be a string"
    assert isinstance(result["documents"], list), "Documents should be a list"
    assert len(result["documents"]) > 0, "Should retrieve at least one document"

def test_dataset_creation():
    """Test dataset creation and loading"""
    dataset = get_or_create_dataset()
    assert dataset is not None, "Dataset should be created or loaded successfully"

# =====================
# MAIN (for manual run)
# =====================
if __name__ == "__main__":
    # Setup dataset
    get_or_create_dataset()
    
    # Run a simple test
    test_rag_bot_single_query()
    print("✅ Basic RAG bot test passed!")
    
    # Run synchronous tests
    test_rag_correctness()
    print("✅ Correctness test passed!")
    
    test_rag_relevance()
    print("✅ Relevance test passed!")
    
    test_rag_groundedness()
    print("✅ Groundedness test passed!")
    
    test_rag_retrieval_relevance()
    print("✅ Retrieval relevance test passed!")
    
    test_rag_all_dimensions()
    print("✅ All dimensions test passed!")
    
    print("🎉 All RAG evaluation tests completed successfully!")