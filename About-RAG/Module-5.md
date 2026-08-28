# 📚 About RAG – Module 5: Retrieval, Prompt Augmentation & Grounding

## 🎯 Objective

This module focuses on understanding how the relevant document chunks retrieved from a vector database are passed to the Large Language Model (LLM) and used to generate a grounded response.

After learning about embeddings, chunking, vector databases, FAISS, and similarity search, the next step is understanding the actual **Retrieval** stage of RAG and how retrieved information is combined with the user's query before being sent to the LLM.

---

## 📖 Topics Covered

- What is a Retriever?
- Role of the Retriever in RAG
- Top-K Retrieval
- How retrieved chunks are passed to the LLM
- Why the LLM should not receive only the user's question
- Prompt Augmentation
- Combining retrieved context with the user's query
- Grounding an LLM response
- How RAG can reduce hallucinations
- Difference between a Retriever and an LLM
- Retrieval as the connection between the knowledge base and the LLM

---

## 🧠 Key Learnings

During this module, I learned that:

- The **Retriever** is responsible for finding and returning the most relevant document chunks for a user's query.
- The retriever uses the query embedding to search the vector database and retrieve the **Top-K most relevant chunks**.
- The retriever does not generate the final answer.
- The **LLM** is responsible for processing the user's question together with the retrieved context and generating the final response.
- Simply sending the user's question to an LLM does not make the system a RAG system because the LLM would not have the retrieved external context.
- **Prompt Augmentation** combines the user's original question with the relevant retrieved context before sending it to the LLM.
- Providing relevant external context helps **ground** the LLM's response in retrieved information.
- RAG can reduce hallucinations by giving the LLM relevant external information instead of making it rely only on its internal knowledge.
- RAG reduces the likelihood of hallucinations, but it does not completely eliminate them.

---

## 🔎 What is a Retriever?

A retriever is the component responsible for retrieving relevant information from the knowledge base based on the user's query.

The process can be represented as:

```text
User Query
     │
     ▼
Query Embedding
     │
     ▼
Vector Database / FAISS
     │
     ▼
Similarity Search
     │
     ▼
Retriever


🔝 Top-K Retrieval

The retriever can return multiple relevant chunks instead of only one.

The number of chunks retrieved is represented by K.

For example:

K = 3
User Query
     │
     ▼
Similarity Search
     │
     ├── Chunk A → Most Relevant
     ├── Chunk B → 2nd Most Relevant
     └── Chunk C → 3rd Most Relevant
     │
     ▼
Relevant Document Chunks



🤔 Why Not Send Only the User's Question to the LLM?

If we send only the user's question to the LLM:

User Question
      │
      ▼
     LLM
      │
      ▼
   Answer

the LLM does not automatically have access to the external documents stored in the RAG knowledge base.

This brings us back to the original limitations of an LLM:

It may not know private information.
It may not know recently added information.
Its knowledge may be limited by its training data.
It may generate incorrect information when it lacks sufficient knowledge.

Therefore, the relevant information retrieved from the knowledge base must be provided to the LLM.

This is where Prompt Augmentation comes into the pipeline.

🧩 Prompt Augmentation

Prompt augmentation is the process of combining the retrieved context with the user's original question before sending it to the LLM.

A simplified prompt can look like:

Context:
Employees may work remotely after completing probation.

Question:
Can employees work from home?

Answer:

The retrieved information provides the context required by the LLM to generate a more relevant response.

The process can be represented as:

Retrieved Context
       │
       │
       ├──────────┐
       │          │
       ▼          ▼
              User Query
                  │
                  ▼
          Prompt Augmentation
                  │
                  ▼
                 LLM
                  │
                  ▼
             Final Answer
🌍 Grounding

Providing retrieved context to the LLM is commonly referred to as grounding the response.

Instead of relying only on information contained within the model, the LLM is given relevant external information retrieved from the knowledge base.

Without RAG:

User Query
     ↓
LLM
     ↓
Answer based mainly on model knowledge

With RAG:

User Query
     ↓
Retrieve Relevant Context
     ↓
Augment Prompt
     ↓
LLM
     ↓
Grounded Answer

The retrieved context gives the LLM a source of relevant information that can be used while generating the response.

🧠 Retriever vs. LLM

The Retriever and LLM have different responsibilities.

Component	Responsibility
Retriever	Finds relevant information from the knowledge base
LLM	Processes the question and retrieved context and generates the final answer

In simple terms:

Retriever → Retrieves
LLM → Generates

The retriever does not answer the question, while the LLM does not independently search the vector database.

Both components work together to complete the RAG pipeline.

⚠️ How RAG Reduces Hallucinations

An LLM can sometimes generate an answer that sounds confident but is incorrect when it does not have sufficient information.

This is known as a hallucination.

RAG helps reduce this problem by retrieving relevant external information and providing it to the LLM as context.

User Query
     │
     ▼
Retrieve Relevant Information
     │
     ▼
Add Context to Prompt
     │
     ▼
LLM
     │
     ▼
Grounded Response

Because the LLM receives relevant context, it has additional information to use when generating its response instead of relying entirely on its internal knowledge.

However:

RAG reduces hallucinations; it does not guarantee that every generated answer will be correct.

Retrieval quality, document quality, chunking, and the LLM itself can still affect the final response.

🔄 Complete Retrieval-to-Generation Flow

The process learned in this module can be represented as:

User Query
     │
     ▼
Query Embedding
     │
     ▼
Vector Database / FAISS
     │
     ▼
Similarity Search
     │
     ▼
Retriever
     │
     ▼
Top-K Relevant Chunks
     │
     ▼
Prompt Augmentation
     │
     ├── Retrieved Context
     │
     └── User Query
            │
            ▼
           LLM
            │
            ▼
      Generated Response

This connects the retrieval stage learned in Module 4 with the generation stage of the RAG pipeline.

💡 Main Insight

The key idea from this module is:

Retrieval finds the relevant information, Prompt Augmentation provides that information to the LLM, and the LLM uses the retrieved context to generate the final response.

The complete idea can be summarized as:

Retrieve
   ↓
Augment
   ↓
Generate

This explains the three core concepts represented by:

Retrieval-Augmented Generation

🏗️ RAG Pipeline So Far

After completing this module, the complete conceptual pipeline is:

Documents
     │
     ▼
Chunking
     │
     ▼
Embeddings
     │
     ▼
Vector Database
     │
     │
     │
User Query
     │
     ▼
Query Embedding
     │
     ▼
Similarity Search
     │
     ▼
Retriever
     │
     ▼
Top-K Relevant Chunks
     │
     ▼
Prompt Augmentation
     │
     ▼
LLM
     │
     ▼
Final Answer
🚀 Next Module

In the next module, I will explore:

How RAG components are implemented using LangChain
Loading documents in code
Text splitting in LangChain
Generating embeddings
Creating a FAISS vector store
Creating a retriever
Connecting the retriever with an LLM
Prompt templates
RetrievalQA
Building the complete RAG pipeline
Mapping the concepts learned so far to actual code

This repository documents my learning journey in Retrieval-Augmented Generation (RAG), progressing from core concepts to implementation.
