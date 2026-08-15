
# 📚 About RAG – Module 2: Embeddings, Semantic Search & Vector Databases

## 🎯 Objective

This module focuses on understanding how RAG systems represent and retrieve information based on **meaning rather than exact keywords**.

The goal is to understand why embeddings are required, how semantic similarity works, and how vector databases enable efficient retrieval of relevant information.

---

## 📖 Topics Covered

* Why keyword-based search is insufficient for RAG
* Introduction to embeddings
* What embeddings represent
* Semantic meaning and similarity
* Vector representations of text
* Similar meanings with different words
* Query embeddings
* Document/chunk embeddings
* Similarity search
* Introduction to vector databases
* Why traditional SQL databases are not ideal for semantic retrieval
* FAISS (Facebook AI Similarity Search)
* Brute-force similarity search vs. efficient similarity search

---

## 🧠 Key Learnings

During this module, I learned that:

* Traditional keyword search primarily looks for matching words, whereas RAG often requires understanding the **semantic meaning** of a query and document.
* Embeddings represent text as **numerical vectors that capture semantic information**.
* Texts with similar meanings tend to have embeddings that are closer together in vector space, even when they use different words.

For example:

```text
User Query:
"Can I work from home?"

Document:
"Employees are permitted to work remotely."

        ↓

Different words
        ↓
Similar meaning
        ↓
Similar embeddings
```

* Both documents and user queries can be converted into embeddings.
* During retrieval, the query embedding is compared with stored document/chunk embeddings to identify the most relevant information.
* A vector database is designed to store embeddings and efficiently perform similarity searches.
* FAISS provides efficient similarity search over vector representations and avoids the need to compare a query against every stored vector using a purely brute-force approach.
* The goal of vector retrieval is not to generate an answer, but to **find the most relevant information that can later be provided to the LLM**.

---

## 🔢 Embeddings in RAG

A simplified representation of the process is:

```text
Document
   │
   ▼
Chunk
   │
   ▼
Embedding Model
   │
   ▼
Numerical Vector
   │
   ▼
Vector Database
```

When a user asks a question:

```text
User Query
   │
   ▼
Query Embedding
   │
   ▼
Similarity Search
   │
   ▼
Most Relevant Chunks
```

The retrieved chunks can then be passed to the next stage of the RAG pipeline.

---

## 🔍 Semantic Search

Semantic search allows a system to retrieve information based on **meaning**, rather than requiring an exact keyword match.

For example:

```text
Query:
"How can I work remotely?"

Document:
"Employees may work from home."

```

Although the wording is different, the meanings are closely related.

Their embeddings should therefore be relatively close in vector space, allowing the relevant document to be retrieved.

---

## 🗄️ Vector Databases

A vector database stores embeddings and supports similarity-based retrieval.

A simplified structure can be represented as:

```text
Document Chunk
      │
      ▼
Embedding
      │
      ▼
Vector Database
      │
      ├── Chunk A → Vector
      ├── Chunk B → Vector
      ├── Chunk C → Vector
      └── Chunk D → Vector
```

When a query arrives, its embedding is compared against the stored vectors to identify the most relevant chunks.

---

## ⚡ FAISS

FAISS stands for:

**Facebook AI Similarity Search**

A naive approach to similarity search would compare the query embedding against every stored embedding one by one.

```text
Query
  │
  ├── Compare → Chunk 1
  ├── Compare → Chunk 2
  ├── Compare → Chunk 3
  ├── Compare → Chunk 4
  └── ...
```

While this can produce accurate results, it becomes inefficient as the number of vectors grows.

FAISS provides optimized methods for similarity search, making retrieval significantly more practical for large collections of embeddings.

---

## 💡 Main Insight

The key idea from this module is:

> **RAG needs to search for meaning, not simply matching words.**

Embeddings provide a numerical representation of semantic meaning, while vector databases provide an efficient way to search those representations.

Therefore:

```text
Text
 ↓
Embedding
 ↓
Vector Database
 ↓
Similarity Search
 ↓
Relevant Information
```

This retrieval process forms the foundation for providing the LLM with the right context.

---

## 🔗 Connection to the RAG Pipeline

After Module 1, the high-level pipeline was:

```text
User Query
      │
      ▼
Retrieve Relevant Information
      │
      ▼
Augment Prompt
      │
      ▼
LLM
      │
      ▼
Response
```

This module explains what happens inside the **retrieval stage**:

```text
User Query
      │
      ▼
Query Embedding
      │
      ▼
Vector Similarity Search
      │
      ▼
Vector Database
      │
      ▼
Relevant Chunks
```

These retrieved chunks will eventually be passed to the LLM as context.

---

## 🚀 Next Module

In the next module, I will explore:

* Document chunking
* Why entire documents should not be embedded as one unit
* Chunk size
* Chunk overlap
* The trade-off between chunks that are too small and too large
* Preserving context during chunking
* How chunking affects retrieval quality

---

*This repository documents my learning journey in Retrieval-Augmented Generation (RAG), progressing from core concepts to implementation.*
