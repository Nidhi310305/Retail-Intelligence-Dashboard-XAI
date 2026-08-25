# 📚 About RAG – Module 4: Vector Databases, FAISS & Similarity Search

## 🎯 Objective

This module focuses on understanding how embedded document chunks are stored and efficiently searched inside a RAG system.

The goal is to understand the role of vector databases, why traditional keyword-based databases are not ideal for semantic retrieval, how similarity search works, and how FAISS enables efficient vector similarity search.

---

## 📖 Topics Covered

* Why embeddings need a specialized storage and search system
* Introduction to vector databases
* Vector databases vs. traditional SQL databases
* Semantic similarity search
* Query embeddings
* Document/chunk embeddings
* Brute-force similarity search
* Introduction to FAISS
* FAISS (Facebook AI Similarity Search)
* Similarity scores and distances
* Finding the most relevant document chunks
* Top-K retrieval
* Role of the vector database in RAG

---

## 🧠 Key Learnings

During this module, I learned that:

* After documents are chunked and converted into embeddings, those embeddings need to be stored in a system optimized for vector similarity search.
* A **vector database** is designed to store embeddings and efficiently retrieve information based on semantic similarity.
* Traditional SQL databases are primarily designed around structured data and exact or rule-based queries, whereas RAG requires searching based on the similarity between numerical vector representations.
* A user's query can also be converted into an embedding.
* The query embedding can then be compared with stored document/chunk embeddings to identify the most relevant information.
* The goal of similarity search is to retrieve the document chunks whose embeddings are most closely related to the query embedding.

---

## 🗄️ Vector Database

A simplified representation of the process is:

```text
Document Chunks
      │
      ▼
Embedding Model
      │
      ▼
Chunk Embeddings
      │
      ▼
Vector Database
```

The vector database stores the embeddings associated with document chunks so that they can later be searched when a user submits a query.

The stored embedding is used for retrieval, while the corresponding **text chunk** provides the actual information that can eventually be passed to the LLM.

---

## 🔍 Why Not Use Only SQL?

Traditional SQL databases are excellent for structured data and exact queries.

For example:

```text
Find employees whose salary > 50,000
```

However, RAG often needs a different type of search:

```text
User:
"Can I work from home?"

Document:
"Employees are permitted to work remotely."
```

The wording is different, but the semantic meaning is similar.

This is why RAG relies on embeddings and vector similarity rather than depending only on exact keyword matching.

> **SQL focuses primarily on structured/exact querying, while vector search enables semantic similarity-based retrieval.**

---

## 🔢 Query Embeddings

The same embedding concept used for document chunks is also applied to the user's query.

```text
User Query
     │
     ▼
Embedding Model
     │
     ▼
Query Embedding
```

The query embedding is then compared with the embeddings stored in the vector database.

For example:

```text
Query Embedding
      │
      ├── Chunk A Embedding → High Similarity
      ├── Chunk B Embedding → Low Similarity
      ├── Chunk C Embedding → Medium Similarity
      └── Chunk D Embedding → Low Similarity
```

The chunks with the highest similarity are considered the most relevant candidates for retrieval.

---

## ⚡ Brute-Force Similarity Search

A simple approach to vector search is to compare the query embedding against every stored embedding.

```text
Query Embedding
      │
      ├── Compare → Chunk 1
      ├── Compare → Chunk 2
      ├── Compare → Chunk 3
      ├── Compare → Chunk 4
      └── ...
```

This approach can work, but as the number of stored embeddings becomes very large, comparing against every vector becomes computationally expensive and inefficient.

This creates the need for optimized similarity-search techniques.

---

## 🚀 FAISS

**FAISS** stands for:

> **Facebook AI Similarity Search**

FAISS is a library designed for efficient similarity search over vector representations.

Instead of relying only on a straightforward comparison of a query against every stored vector, FAISS provides optimized indexing and search methods for working with large collections of embeddings.

A simplified view is:

```text
Query Embedding
      │
      ▼
     FAISS
      │
      ▼
Similarity Search
      │
      ▼
Most Relevant Vectors
      │
      ▼
Corresponding Document Chunks
```

FAISS does **not** generate the final natural-language answer.

Its responsibility is to help locate the relevant information.

---

## 📊 Similarity Search

The purpose of similarity search is to determine which stored embeddings are closest or most similar to the query embedding.

For example:

```text
Query:
"Can employees work remotely?"

             Similarity
Chunk A  ─────── 0.96  ← Most Relevant
Chunk B  ─────── 0.72
Chunk C  ─────── 0.31
Chunk D  ─────── 0.15
```

The exact similarity calculation depends on the chosen similarity or distance measure and indexing approach.

The important concept is:

> **The query embedding is compared with stored embeddings to find the most relevant information.**

---

## 🔝 Top-K Retrieval

Instead of retrieving only one result, a RAG system can retrieve the **Top-K** most relevant chunks.

For example, if:

```text
K = 3
```

the system retrieves the three highest-ranked relevant chunks.

```text
Query
  │
  ▼
Similarity Search
  │
  ├── Chunk A → Highest similarity
  ├── Chunk B → 2nd highest
  └── Chunk C → 3rd highest
```

These retrieved chunks can then be provided as context for the next stage of the RAG pipeline.

---

## 🔄 Connection to the RAG Pipeline

At this point, the RAG pipeline can be represented as:

```text
Documents
     │
     ▼
Chunking
     │
     ▼
Chunk Embeddings
     │
     ▼
Vector Database / FAISS
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
Top-K Relevant Chunks
```

The retrieved chunks are now ready for the next stage: providing relevant context to the LLM.

---

## 💡 Main Insight

The key idea from this module is:

> **Embeddings make semantic representation possible, while vector databases and similarity-search systems make that representation searchable.**

The vector database does not generate the final response.

Its role is to efficiently locate the information that is most relevant to the user's query.

Therefore:

```text
Query
  ↓
Query Embedding
  ↓
Similarity Search
  ↓
Relevant Chunk(s)
  ↓
LLM Context
```

This retrieval mechanism forms the core of the **Retrieval** part of Retrieval-Augmented Generation.

---

## 🚀 Next Module

In the next module, I will explore:

* Retrievers
* Top-K retrieval in practice
* How retrieved chunks are passed forward
* Prompt augmentation
* Combining retrieved context with the user's query
* Grounding an LLM response using retrieved information
* How retrieval helps reduce hallucinations

---

*This repository documents my learning journey in Retrieval-Augmented Generation (RAG), progressing from core concepts to implementation.*
