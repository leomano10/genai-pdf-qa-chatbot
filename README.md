# Development of a PDF-Based Question-Answering Chatbot Using LangChain

## AIM

To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain and Gemini API, and to evaluate its effectiveness by testing its responses to different queries derived from the document's content.

---

# PROBLEM STATEMENT

Develop a PDF-Based Question-Answering Chatbot capable of reading and processing a PDF document and answering user questions based on the content of that document.

The system follows the Retrieval-Augmented Generation (RAG) approach.

The PDF content is:

1. Loaded and processed.
2. Split into smaller text chunks.
3. Converted into vector embeddings.
4. Stored in a FAISS vector database.

When a user asks a question, the system retrieves the most relevant information from the PDF and sends the retrieved context along with the question to the Gemini model.

The chatbot generates an answer based only on the information available in the provided PDF.

---

# DESIGN STEPS

## STEP 1: Load and Process the PDF

The PDF document is loaded using `PyPDFLoader` from LangChain.

The extracted text is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

Configuration used:

- Chunk Size: `1000`
- Chunk Overlap: `200`

This makes the document easier to search and retrieve relevant information.

---

## STEP 2: Generate Embeddings and Create Vector Database

The text chunks are converted into numerical vector representations using the HuggingFace embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The generated embeddings are stored in a FAISS vector database.

When a user asks a question, FAISS performs similarity search and retrieves the most relevant text chunks from the PDF.

---

## STEP 3: Generate Answers Using Gemini

The relevant text chunks retrieved from the FAISS vector database are provided as context to the Gemini model.

The chatbot follows these rules:

1. Answer only using the provided PDF context.
2. Do not use outside knowledge.
3. Do not generate false information.
4. If the answer is unavailable, clearly inform the user.
5. Display the source pages used to answer the question.

---

# SYSTEM ARCHITECTURE

```text
PDF FILE
   │
   ▼
PyPDFLoader
   │
   ▼
Text Extraction
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ▼
Text Chunks
   │
   ▼
HuggingFace Embeddings
(all-MiniLM-L6-v2)
   │
   ▼
FAISS Vector Database
   │
User Question
   │
   ▼
Similarity Search
   │
   ▼
Relevant PDF Context
   │
   ▼
Gemini API
   │
   ▼
Final Answer
```

---

# PROGRAM

## Full Python Code

```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ============================================================
# PDF-BASED QUESTION ANSWERING CHATBOT USING LANGCHAIN + GEMINI
# ============================================================

# ------------------------------------------------------------
# 1. INSTALL REQUIRED LIBRARIES
# ------------------------------------------------------------

!pip install -q -U google-genai langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu pypdf sentence-transformers


# ------------------------------------------------------------
# 2. IMPORT LIBRARIES
# ------------------------------------------------------------

import os
from getpass import getpass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from google import genai

print("All libraries imported successfully!")


# ------------------------------------------------------------
# 3. SET GEMINI API KEY SECURELY
# ------------------------------------------------------------

if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass(
        "Enter your Gemini API Key: "
    )

print("Gemini API key configured successfully!")


# ------------------------------------------------------------
# 4. INITIALIZE GEMINI CLIENT
# ------------------------------------------------------------

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

print("Gemini client initialized successfully!")


# ------------------------------------------------------------
# 5. DISPLAY AVAILABLE GEMINI MODELS
# ------------------------------------------------------------

print("\nChecking available Gemini models...\n")

available_models = []

try:

    for model in client.models.list():

        model_name = getattr(
            model,
            "name",
            None
        )

        if model_name:

            available_models.append(
                model_name
            )

            print(
                model_name
            )

except Exception as e:

    print(
        "Could not list models."
    )

    print(
        "Reason:",
        e
    )


# ------------------------------------------------------------
# 6. SELECT GEMINI MODEL
# ------------------------------------------------------------

preferred_models = [

    "gemini-3.7-flash",

    "gemini-3.6-flash",

    "gemini-3.5-flash",

    "gemini-2.5-flash",

    "gemini-2.5-flash-lite"

]

MODEL_NAME = None


for preferred_model in preferred_models:

    if (

        preferred_model
        in available_models

        or

        f"models/{preferred_model}"
        in available_models

    ):

        MODEL_NAME = (
            preferred_model
        )

        break


if MODEL_NAME is None:

    MODEL_NAME = (
        "gemini-2.5-flash"
    )

    print(
        "\nAutomatic model detection failed."
    )

    print(
        "Using fallback model:",
        MODEL_NAME
    )

else:

    print(
        "\nSelected Gemini model:",
        MODEL_NAME
    )


# ------------------------------------------------------------
# 7. SPECIFY PDF FILE PATH
# ------------------------------------------------------------

pdf_path = "sem2result.pdf"


# ------------------------------------------------------------
# 8. CHECK WHETHER PDF EXISTS
# ------------------------------------------------------------

if not os.path.exists(
    pdf_path
):

    print(
        "\nERROR: PDF FILE NOT FOUND!"
    )

    print(
        "\nCurrent working directory:"
    )

    print(
        os.getcwd()
    )

    print(
        "\nFiles available:"
    )

    for file in os.listdir():

        print(
            file
        )

    raise FileNotFoundError(

        f"\nPDF file '{pdf_path}' "
        "was not found."
        "\nPlease place the PDF "
        "in the same folder as "
        "the Jupyter Notebook."

    )


print(
    "\nPDF found successfully!"
)

print(
    "PDF Path:",
    pdf_path
)


# ------------------------------------------------------------
# 9. LOAD PDF
# ------------------------------------------------------------

print(
    "\nLoading PDF..."
)

loader = PyPDFLoader(
    pdf_path
)

documents = loader.load()

print(
    "PDF loaded successfully!"
)

print(
    "Number of pages:",
    len(documents)
)


# ------------------------------------------------------------
# 10. DISPLAY SAMPLE CONTENT
# ------------------------------------------------------------

if len(documents) > 0:

    print(
        "\n" + "=" * 70
    )

    print(
        "SAMPLE CONTENT FROM PDF"
    )

    print(
        "=" * 70
    )

    print(
        documents[0].page_content[:1000]
    )

    print(
        "\n" + "=" * 70
    )


# ------------------------------------------------------------
# 11. SPLIT DOCUMENT INTO CHUNKS
# ------------------------------------------------------------

print(
    "\nSplitting document into chunks..."
)

text_splitter = (
    RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    )
)


chunks = (
    text_splitter.split_documents(
        documents
    )
)


print(
    "Document split successfully!"
)

print(
    "Number of chunks:",
    len(chunks)
)


# ------------------------------------------------------------
# 12. LOAD EMBEDDING MODEL
# ------------------------------------------------------------

print(
    "\nLoading embedding model..."
)

embedding_model = (
    HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"

    )
)


print(
    "Embedding model loaded successfully!"
)


# ------------------------------------------------------------
# 13. CREATE FAISS VECTOR DATABASE
# ------------------------------------------------------------

print(
    "\nCreating FAISS vector database..."
)


vector_store = (
    FAISS.from_documents(

        documents=chunks,

        embedding=embedding_model

    )
)


print(
    "FAISS vector database created successfully!"
)


# ------------------------------------------------------------
# 14. CREATE RETRIEVER FUNCTION
# ------------------------------------------------------------

def get_relevant_context(
    question,
    k=4
):

    relevant_documents = (
        vector_store.similarity_search(

            query=question,

            k=k

        )
    )


    context_parts = []


    for i, doc in enumerate(
        relevant_documents
    ):

        page_number = (
            doc.metadata.get(
                "page",
                "Unknown"
            )
        )


        # PyPDFLoader starts pages from 0
        if isinstance(
            page_number,
            int
        ):

            page_number += 1


        context_parts.append(

            f"""
SOURCE {i + 1}

PAGE: {page_number}

CONTENT:

{doc.page_content}
"""

        )


    context = "\n\n".join(
        context_parts
    )


    return (

        context,

        relevant_documents

    )


print(
    "Retriever function created successfully!"
)


# ------------------------------------------------------------
# 15. CREATE QUESTION-ANSWERING FUNCTION
# ------------------------------------------------------------

def ask_pdf(question):


    context, relevant_documents = (
        get_relevant_context(

            question,

            k=4

        )
    )


    prompt = f"""

You are a PDF-based Question Answering Chatbot.

Your task is to answer the user's question using ONLY
the information provided in the PDF context below.

IMPORTANT RULES:

1. Answer only from the provided context.

2. Do not use outside knowledge.

3. Do not invent information.

4. If the answer cannot be found in the context,
say exactly:

"I could not find the answer in the provided PDF."

5. Give a clear and concise answer.

6. Use relevant information from the document.

--------------------------------------------------

PDF CONTEXT:

{context}

--------------------------------------------------

USER QUESTION:

{question}

--------------------------------------------------

ANSWER:

"""


    try:


        response = (
            client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt

            )
        )


        answer = getattr(

            response,

            "text",

            None

        )


        if not answer:

            answer = (
                "The model returned an empty response."
            )


        return (

            answer,

            relevant_documents

        )


    except Exception as e:


        error_message = f"""

Error while generating answer.

Model used:

{MODEL_NAME}

Error:

{str(e)}

Try using one of the models displayed
in the available Gemini models list.

"""


        return (

            error_message,

            relevant_documents

        )


print(
    "Question-answering chatbot created successfully!"
)


# ------------------------------------------------------------
# 16. TEST THE CHATBOT
# ------------------------------------------------------------

test_question = (
    "What is the main topic of this document?"
)


print(
    "\n" + "=" * 70
)

print(
    "TESTING THE CHATBOT"
)

print(
    "=" * 70
)


print(
    "\nQUESTION:"
)

print(
    test_question
)


answer, sources = ask_pdf(
    test_question
)


print(
    "\nANSWER:"
)

print(
    answer
)


# ------------------------------------------------------------
# 17. DISPLAY SOURCE DOCUMENTS
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "SOURCE DOCUMENTS USED"
)

print(
    "=" * 70
)


for i, doc in enumerate(
    sources
):


    page_number = (
        doc.metadata.get(

            "page",

            "Unknown"

        )
    )


    if isinstance(
        page_number,
        int
    ):

        page_number += 1


    print(
        f"\nSOURCE {i + 1}"
    )


    print(
        "PDF Page:",
        page_number
    )


    print(
        "\nContent Preview:"
    )


    print(
        doc.page_content[:500]
    )


    print(
        "\n" + "-" * 60
    )


# ------------------------------------------------------------
# 18. START INTERACTIVE CHATBOT
# ------------------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "PDF QUESTION-ANSWERING CHATBOT"
)

print(
    "=" * 70
)


print(
    "\nAsk questions about your PDF."
)


print(
    "Type 'exit' or 'quit' to stop."
)


while True:


    user_question = input(
        "\nYou: "
    )


    # Exit chatbot
    if user_question.lower().strip() in [

        "exit",

        "quit"

    ]:


        print(
            "\nChatbot: Goodbye!"
        )

        break


    # Ignore empty questions
    if not user_question.strip():


        print(
            "\nChatbot: Please enter a question."
        )

        continue


    # Generate answer
    answer, sources = ask_pdf(
        user_question
    )


    print(
        "\nChatbot:"
    )


    print(
        answer
    )


    print(
        "\nSources used:"
    )


    if sources:


        for i, doc in enumerate(
            sources
        ):


            page_number = (
                doc.metadata.get(

                    "page",

                    "Unknown"

                )
            )


            if isinstance(
                page_number,
                int
            ):

                page_number += 1


            print(

                f"Source {i + 1}: "
                f"PDF Page {page_number}"

            )


    else:


        print(
            "No source documents found."
        )


    print(
        "\n" + "-" * 70
    )
```

---

# HOW TO RUN

## Step 1: Install Required Libraries

The notebook automatically installs the required libraries.

The libraries used are:

```text
google-genai
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
faiss-cpu
pypdf
sentence-transformers
```

---

## Step 2: Add the PDF File

Place your PDF file in the same folder as the Jupyter Notebook.

Example:

```text
PDF-Based-QA-Chatbot/
│
├── PDF_Based_QA_Chatbot.ipynb
├── sem2result.pdf
├── README.md
└── .gitignore
```

---

## Step 3: Run the Notebook

Run all the cells.

The program will ask for your Gemini API key securely:

```text
Enter your Gemini API Key:
```

The API key will not be printed in the output.

---

# SAMPLE QUESTIONS

You can test the chatbot using questions such as:

```text
What is the main topic of this document?
```

```text
What important information is mentioned in the document?
```

```text
Summarize the document in simple words.
```

For a semester result PDF:

```text
What subjects and marks are mentioned in the document?
```

```text
What is the overall result mentioned in this document?
```

---

# OUTPUT

The chatbot retrieves relevant information from the PDF and generates an answer based on the retrieved context.

The program also displays the PDF pages used to answer the question.

Example:

```text
QUESTION:

What is the main topic of this document?

ANSWER:

The chatbot retrieves relevant information from the PDF
and generates an answer based on the provided document context.

Sources used:

Source 1: PDF Page 1
Source 2: PDF Page 2
Source 3: PDF Page 3
```

---

# RESULT

A PDF-Based Question-Answering Chatbot was successfully developed using LangChain, HuggingFace Embeddings, FAISS Vector Database, and the Gemini API.

The system successfully:

- Loads and processes PDF documents.
- Extracts text from PDF files.
- Splits PDF content into smaller chunks.
- Generates vector embeddings.
- Stores embeddings in a FAISS vector database.
- Performs semantic similarity search.
- Retrieves relevant information based on user questions.
- Uses Gemini to generate context-aware answers.
- Restricts answers to information available in the PDF.
- Displays source pages used for answering questions.
- Provides an interactive chatbot interface.

Thus, the aim of developing a PDF-Based Question-Answering Chatbot using LangChain was successfully achieved.

---

# TECHNOLOGIES USED

- Python
- Jupyter Notebook
- LangChain
- Google Gemini API
- HuggingFace
- Sentence Transformers
- FAISS
- PyPDF

---

# PROJECT STRUCTURE

```text
PDF-Based-QA-Chatbot/
│
├── README.md
├── PDF_Based_QA_Chatbot.ipynb
├── sem2result.pdf
└── .gitignore
```

---

# SECURITY NOTE

Do not upload your Gemini API key to GitHub.

The program securely requests the API key using:

```python
from getpass import getpass

if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass(
        "Enter your Gemini API Key: "
    )
```

Never hardcode your Gemini API key in the notebook or GitHub repository.

Add the following to your `.gitignore` file:

```text
.env
*.env
__pycache__/
.ipynb_checkpoints/
```

---

# AUTHOR

**Name:** Manorajapriyan

**Department:** B.E. Computer Science and Engineering

**Institution:** Saveetha Engineering College
