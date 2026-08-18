# 📚 About RAG – Module 3: Document Chunking

## 🎯 Objective

This module focuses on understanding **document chunking**, an essential preprocessing step in a Retrieval-Augmented Generation (RAG) system.

The goal is to understand why large documents should be divided into smaller, meaningful sections before generating embeddings, how chunk size affects retrieval, and how chunk overlap helps preserve context across boundaries.

---

## 📖 Topics Covered

* What is document chunking?
* Why large documents should be split into smaller chunks
* Why an entire document should not be represented by a single embedding
* Relationship between chunking and embeddings
* Chunk size
* Chunk overlap
* Why semantic meaning should be preserved within chunks
* Problems with chunks that are too small
* Problems with chunks that are too large
* Choosing an appropriate chunk size
* How chunking affects retrieval quality

---

## 🧠 Key Learnings

During this module, I learned that:

* Large documents such as PDFs may contain hundreds or thousands of pages and can contain many different topics.
* Creating one embedding for an entire document can mix unrelated information and make retrieval less precise.
* **Chunking** is the process of dividing a large document into smaller, manageable pieces of text.
* Each chunk can then be converted into its own embedding and stored in a vector database.
* Smaller, focused chunks allow the retrieval system to identify more specific and relevant information.

The basic process is:

```text
Large Document
      │
      ▼
Document Chunking
      │
      ├── Chunk 1
      ├── Chunk 2
      ├── Chunk 3
      └── ...
      │
      ▼
Embeddings
```

---

## 📏 Chunk Size

Chunk size determines how much text is included in each chunk.

For example:

```text
chunk_size = 500
```

means that a chunk can contain up to approximately 500 characters, depending on the text-splitting method being used.

The choice of chunk size involves a trade-off.

### Chunks that are too small

Very small chunks may:

* Lose important surrounding context
* Split related information apart
* Make individual chunks less meaningful

### Chunks that are too large

Very large chunks may:

* Contain unrelated information
* Make retrieval less precise
* Provide unnecessary context to the LLM
* Consume more of the available context window

Therefore, the goal is to find a **balanced chunk size** that preserves enough meaning without including excessive unrelated information.

---

## 🔗 Chunk Overlap

Chunk overlap allows a portion of one chunk to be repeated in the following chunk.

For example:

```text
Chunk 1:
[--------------------500 characters--------------------]

                    ↓ 100 characters overlap

Chunk 2:
             [--------------------500 characters--------------------]
```

With:

```text
chunk_size = 500
chunk_overlap = 100
```

approximately 100 characters from the end of one chunk are carried into the next chunk.

### Why is overlap useful?

Important information can sometimes fall exactly at a chunk boundary.

Without overlap:

```text
Chunk 1 → Beginning of an important idea

Chunk 2 → Remaining part of the idea
```

The meaning may become fragmented.

With overlap:

```text
Chunk 1 → Beginning + part of the idea

Chunk 2 → Repeated context + continuation
```

This helps preserve **semantic continuity** across chunk boundaries and can improve retrieval quality.

---

## ⚖️ The Chunking Trade-off

There is no single chunk size that works perfectly for every RAG application.

The appropriate size depends on factors such as:

* Type of documents
* Structure of the information
* Length of individual concepts
* Retrieval requirements
* Context available to the LLM

For example, legal documents may require larger chunks to preserve complete clauses, while shorter informational documents may work well with smaller chunks.

The key principle is:

> **Chunks should be large enough to preserve meaning but small enough to remain focused and retrievable.**

---

## 🔄 Chunking and Embeddings

Chunking happens **before** embedding.

The process is:

```text
Document
    │
    ▼
Chunking
    │
    ▼
Individual Chunks
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

This means the vector database does not simply store one representation for an entire PDF.

Instead, it can contain many embeddings corresponding to different chunks of the document.

This allows the retrieval system to find the specific section that is most relevant to a user's query.

---

## 💡 Main Insight

The key idea from this module is:

> **Good retrieval depends on good chunking.**

If the document is divided poorly, even a powerful embedding model and vector database may retrieve incomplete or irrelevant information.

Therefore, chunking is not simply about splitting text into arbitrary pieces.

It is about creating **meaningful, focused units of information** that can later be embedded and retrieved effectively.

---

## 🔗 Connection to the RAG Pipeline

The RAG pipeline can now be understood in greater detail:

```text
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
     ▼
Similarity Search
     │
     ▼
Relevant Chunks
```

Chunking therefore acts as the bridge between the original documents and the embedding/retrieval process.

---

## 🚀 Next Module

In the next module, I will explore:

* Vector databases in greater depth
* FAISS
* How similarity search works
* Query embeddings
* Comparing query and document embeddings
* Top-K retrieval
* How relevant chunks are selected from the vector database

---

*This repository documents my learning journey in Retrieval-Augmented Generation (RAG), progressing from core concepts to implementation.*
