# FinTech Compliance Explainer Bot

A chatbot that helps users understand FinTech, payment systems, and compliance concepts through simple, AI-generated explanations grounded in reference documents.

## Problem Statement

Financial and payment compliance information can be difficult to understand because it is often spread across different sources and written in technical language.

The FinTech Compliance Explainer Bot aims to make this information more accessible by allowing users to ask questions in plain English and receive understandable explanations.

## How It Works

The project uses Retrieval-Augmented Generation (RAG) to combine document retrieval with AI-generated responses.

1. **Document Loading:** Loads reference documents containing relevant information.
2. **Text Chunking:** Splits documents into smaller sections for efficient retrieval.
3. **Embeddings:** Converts text chunks into numerical representations using a Hugging Face embedding model.
4. **Vector Search:** Uses FAISS to retrieve document chunks relevant to the user's question.
5. **Response Generation:** Uses Google's Gemini model to generate an explanation based on the question and retrieved context.
6. **Answer Delivery:** Returns the generated explanation to the user through the application interface.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Google Gemini | Natural-language response generation |
| LangChain | Integrating components of the RAG pipeline |
| Hugging Face | Text embeddings |
| FAISS | Similarity search and document retrieval |
| Streamlit | User interface, if used in the application |

## Project Structure

- `rag_pipeline.py`: Implements document processing, retrieval, and response generation.
- `requirements.txt`: Lists the project's Python dependencies.
- `test_retrieval.py`: Contains retrieval-related tests.
- `test_project_setup.py`: Contains project setup tests.
- `dl.txt`, `genai.txt`, `india.txt`, `lr.txt`, `ml.txt`, `usa.txt`: Reference text files for the knowledge base.
- `README.md`: Project overview and setup instructions.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rupikagouri/Fintech-Compliance-Explainer-Bot.git
cd Fintech-Compliance-Explainer-Bot
