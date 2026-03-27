import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")
    

def get_related_data(query):
  embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))

  store = PGVector(
      embeddings=embeddings,
      collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
      connection=os.getenv("DATABASE_URL"),
      use_jsonb=True
  )

  results = store.similarity_search_with_score(query=query, k=10)
  concatenated_results = ""

  for i, (doc, score) in enumerate(results, start=1):
      concatenated_results += f"-"*50 + "\n"
      concatenated_results += f"Resultado {i} ; Score: {score}\n"
      concatenated_results += f"-"*50 + "\n"
      concatenated_results += "Texto:\n"
      concatenated_results += f"Content: {doc.page_content.strip()}\n"
  
  return concatenated_results

def invoke_model_with_data(query, data):
  template_search = PromptTemplate(
      input_variables=["contexto", "pergunta"],
      template=PROMPT_TEMPLATE,
  )

  gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

  pipeline = template_search | gemini | StrOutputParser()
  result = pipeline.invoke({
    "contexto": data,
    "pergunta": query
  })

  return result

def search_prompt(question=None):
  if not question:
    return None
  
  data = get_related_data(question)
  return invoke_model_with_data(question, data)