# 📚 Desafio MBA: Ingestão e Busca Semântica com LangChain e Postgres

Um projeto que implementa um sistema de RAG (Retrieval Augmented Generation) para ingerir documentos PDF e realizar buscas semânticas com LLM usando LangChain, PostgreSQL com pgvector e Google Gemini.

## 🎯 Objetivos

Este desafio tem como objetivo desenvolver um sistema completo de:

1. **Ingestão de PDF** - Carregar e processar documentos PDF com divisão inteligente de chunks
2. **Busca Semântica** - Implementar busca vetorial em PostgreSQL com pgvector
3. **Augmented Generation** - Usar LLM para gerar respostas baseadas apenas no contexto do documento
4. **Interface CLI** - Criar uma linha de comando para interagir com o sistema

## 🛠️ Tecnologias Obrigatórias

- **Linguagem**: Python
- **Framework RAG**: LangChain
- **Banco de Dados**: PostgreSQL + pgvector
- **Modelo de Embedding**: Google Generative AI
- **Modelo LLM**: Google Gemini
- **Containerização**: Docker & Docker Compose

## 📋 Pré-requisitos

- Python 3.8+
- Docker e Docker Compose instalados
- Conta Google Cloud com Google Generative AI API habilitada
- pip (gerenciador de pacotes Python)

## 🚀 Instalação e Configuração

### 1. Clonar o Repositório e Navegar para o Diretório

```bash
cd mba-ia-desafio-ingestao-busca
```

### 2. Criar Ambiente Virtual

#### No Bash/Git Bash:
```bash
python -m venv venv
source venv/Scripts/activate
```

#### No PowerShell:
```powershell
python -m venv venv
venv/Scripts/activate.ps1
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e preencha as seguintes variáveis:

```env
# Chave de API do Google Cloud
GOOGLE_API_KEY=sua_chave_api_google_aqui

# Modelo de embedding (use models/gemini-embedding-001 ou models/embedding-001)
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001

# URL de conexão do PostgreSQL
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Nome da coleção para armazenar os vetores
PG_VECTOR_COLLECTION_NAME=seu_nome_colecao

# Caminho para o arquivo PDF a ser processado
PDF_PATH=path/to/seu/documento.pdf
```

#### Como obter a chave de API do Google:

1. Acesse [Google Cloud Console](https://console.cloud.google.com/)
2. Crie um novo projeto
3. Ative a API "Generative Language API"
4. Crie uma chave de API (Application default credentials)
5. Copie a chave para a variável `GOOGLE_API_KEY` no `.env`

### 5. Inicializar o Banco de Dados

```bash
docker compose up -d
```

Este comando irá:
- Iniciar o PostgreSQL 17 com pgvector habilitado no Docker
- Criar os volumes necessários para persistência de dados
- Realizar health checks automáticos

Verifique se está rodando:
```bash
docker compose ps
```

## 📖 Como Usar a Aplicação

### 1. Ingerir um Documento PDF

Execute o script de ingestão para processar um PDF e armazená-lo no banco de dados:

```bash
python src/ingest.py
```

**O que acontece:**
- Carrega o PDF especificado em `PDF_PATH`
- Divide o documento em chunks de 1000 caracteres com 150 caracteres de sobreposição
- Gera embeddings vetoriais para cada chunk usando Google Gemini
- Armazena os vetores no PostgreSQL com metadados

⚠️ **Nota**: A ingestão consome tokens de API. Se você tiver limites de taxa, descomente o código comentado em `save_documents()` para adicionar delay entre requisições.

### 2. Fazer Perguntas (Chat)

Após ingerir o PDF, você pode fazer perguntas sobre o conteúdo:

#### Via argumento CLI:
```bash
python src/chat.py "Qual é o tema principal do documento?"
```

#### Via entrada interativa:
```bash
python src/chat.py
# Digite sua pergunta e pressione Enter
```

**Como funciona:**
1. Converte a pergunta em um vetor semântico
2. Busca os 10 chunks mais similares no banco de dados
3. Envia o contexto + pergunta para o Gemini
4. O LLM gera uma resposta baseada APENAS no contexto do documento

### 3. Fazer Buscas Semânticas Diretas

Para usar a funcionalidade de busca sem o LLM:

```python
from src.search import get_related_data

resultados = get_related_data("sua pergunta aqui")
print(resultados)
```

## 📁 Estrutura do Projeto

```
mba-ia-desafio-ingestao-busca/
├── src/
│   ├── ingest.py      # Script de ingestão de PDFs e armazenamento em PostgreSQL
│   ├── search.py      # Lógica de busca semântica e integração com LLM
│   ├── chat.py        # Interface CLI para perguntar ao sistema
│   └── __pycache__/   # Cache de compilação Python
├── docker-compose.yml # Configuração do PostgreSQL + pgvector
├── requirements.txt   # Dependências Python
├── .env              # Variáveis de ambiente (NÃO COMMITAR)
├── .env.example      # Template de variáveis de ambiente
└── README.md         # Este arquivo
```

### Descrição dos Arquivos

- **ingest.py**: Responsável por carregar PDFs, dividir em chunks e armazenar vetores no PostgreSQL
- **search.py**: Implementa a busca semântica e integração com o LLM Gemini
- **chat.py**: Interface de linha de comando para fazer perguntas

## 🔧 Configurações Importantes

### Tamanho dos Chunks de Documento

Em `src/ingest.py`, você pode ajustar:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Tamanho de cada chunk
    chunk_overlap=150,      # Sobreposição entre chunks
    add_start_index=False
)
```

### Número de Resultados da Busca

Em `src/search.py`, ajuste o parâmetro `k`:

```python
results = store.similarity_search_with_score(query=query, k=10)
```

### Modelo de Embedding

Você pode usar diferentes modelos:
- `models/gemini-embedding-001` (recomendado)
- `models/embedding-001`

### Configuração do LLM

Em `src/search.py`:

```python
gemini = init_chat_model(
    "gemini-2.5-flash",    # Modelo
    model_provider="google_genai",
    temperature=0            # 0 = determinístico, 1 = criativo
)
```

## 🐳 Gerenciar Docker

### Ver status dos containers:
```bash
docker compose ps
```

### Ver logs do PostgreSQL:
```bash
docker compose logs postgres
```

### Parar os containers:
```bash
docker compose down
```

### Remover volumes (limpar dados):
```bash
docker compose down -v
```

### Conectar ao PostgreSQL diretamente:
```bash
docker compose exec postgres psql -U postgres -d rag
```

## ⚠️ Limitações e Considerações

1. **Limites de Taxa da Api**: Google Generative AI tem limites de requisições. Para pequenos testes, use sem delay. Para produção, considere adicionar delays ou usar batchs menores.

2. **Tamanho máximo de contexto**: O LLM tem limite de tokens. Se o documento for muito grande, considere aumentar o `chunk_overlap`.

3. **Segurança da API Key**: Nunca commite o arquivo `.env` com suas chaves. Use `.gitignore` para excluir.

4. **Qualidade das Respostas**: Depende da qualidade dos embeddings e da relevância dos chunks recuperados. Ajuste `k` e `chunk_size` conforme necessário.

## 🔒 Segurança

- ✅ O arquivo `.env` está listado em `.gitignore`
- ✅ Use `.env.example` como template
- ✅ Nunca compartilhe suas chaves de API
- ✅ Regenere chaves se forem expostas

## 📚 Pacotes Utilizados

- **langchain**: Framework de RAG e orquestração de LLMs
- **langchain-postgres**: Integração PostgreSQL com LangChain
- **langchain-google-genai**: Suporte para Google Generative AI
- **langchain-text-splitters**: Divisão inteligente de textos
- **python-dotenv**: Carregamento de variáveis de ambiente
- **psycopg**: Driver PostgreSQL assíncrono
- **asyncpg**: Adaptador PostgreSQL async

## 🐛 Troubleshooting

### Erro: "Environment variable not set"
**Solução**: Verifique se o arquivo `.env` existe e tem todas as variáveis preenchidas.

### Erro: "Connection refused" ao conectar ao PostgreSQL
**Solução**: Certifique-se de que o Docker está rodando e o container está healthy:
```bash
docker compose up -d
docker compose ps
```

### Erro: "GOOGLE_API_KEY invalid"
**Solução**: Verifique se sua chave de API está correta e se a API está habilitada no Google Cloud Console.

### Erro: "File not found" para o PDF
**Solução**: Verifique o caminho do arquivo PDF na variável `PDF_PATH`. Use caminhos absolutos ou relativos corretamente.

### Performance lenta durante a ingestão
**Solução**: Aumente o delay ou reduza o tamanho dos chunks para menos requisições à API.

## 📖 Referências

- [LangChain Documentation](https://python.langchain.com/)
- [PostgreSQL pgvector Extension](https://github.com/pgvector/pgvector)
- [Google Generative AI Python SDK](https://github.com/googleapis/python-genai)
- [Docker Documentation](https://docs.docker.com/)

## 📝 Notas Finais

Este projeto é parte do desafio do MBA em Engenharia de Software com IA da Full Cycle. Ele demonstra conceitos práticos de:
- Retrieval Augmented Generation (RAG)
- Vector Databases
- LLM Integration
- Document Processing

---

**Desenvolvido como desafio técnico do MBA IA - Full Cycle**