/*
📌 Purpose – Outlines the phased development plan for the Video Timeline Analyzer, centered on the variable-granularity DataFrame and reproducible metadata alignment.
🔄 Latest Changes – Initial creation for the de-novo branch, reflecting the new architecture and scientific best practices.
⚙️ Key Logic – Defines project phases, milestones, and review points, with a focus on modular pipeline development, metadata alignment, and reproducibility.
📂 Expected File Path – docs/ROADMAP.md
🧠 Reasoning – Ensures structured, transparent, and reproducible project progression, with the DataFrame as the canonical metadata source.
*/

# Project Roadmap (De-Novo Branch)

## Overview

This roadmap defines the major phases, milestones, and review points for the Video Timeline Analyzer, with a focus on the variable-granularity DataFrame as the central metadata structure. All pipeline outputs are parsed and aligned into this DataFrame, which is then used for Qdrant ingestion and downstream analysis.

---

## Phase 1: Documentation & Planning
- [ ] Finalize system architecture ([ARCHITECTURE.md](ARCHITECTURE.md)), emphasizing the DataFrame-centric design
- [ ] Complete technical specifications ([SPECIFICATIONS.md](SPECIFICATIONS.md))
- [ ] Define development and environment setup ([DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md))
- [ ] Establish contribution and review guidelines ([DEVELOPMENT.md](DEVELOPMENT.md))
- [ ] Review and approve all documentation

**Milestone:** All documentation reviewed and approved by project stakeholders.

---

## Phase 2: Modular Pipeline Implementation
- [ ] Implement scene detection module (parameterized, reproducible)
- [ ] Implement audio extraction and transcription module (Whisper, parameterized segment length)
- [ ] Implement key frame extraction and image embedding module (CLIP or similar)
- [ ] Validate each module independently with test videos

**Milestone:** All core modules implemented and tested in isolation.

---

## Phase 3: Metadata Alignment & DataFrame Construction
- [ ] Design and implement alignment logic for variable-granularity metadata (scenes, frames, transcripts)
- [ ] Build canonical DataFrame structure (one row per scene, lists for variable fields)
- [ ] Parameterize and document all alignment steps
- [ ] Persist DataFrame (e.g., Parquet) for reproducibility
- [ ] Validate DataFrame content and alignment with test cases

**Milestone:** Canonical DataFrame construction and validation complete.

---

## Phase 4: Qdrant Integration & Querying
- [ ] Implement Qdrant ingestion (multi-vector support, full payloads)
- [ ] Validate round-trip querying (text, image, metadata)
- [ ] Document Qdrant schema and ingestion process

**Milestone:** Qdrant collection populated and queryable with all metadata.

---

## Phase 5: Scientific Validation & Extensibility
- [ ] Benchmark pipeline reproducibility and performance
- [ ] Add support for additional metadata fields (tags, scores, etc.) as needed
- [ ] Document all parameters, random seeds, and alignment logic
- [ ] Prepare for publication, sharing, or further extension (e.g., plugin system, distributed processing)

**Milestone:** System meets scientific standards for reproducibility, extensibility, and documentation.

---

## Review & Feedback
- Regular review meetings at the end of each phase
- Issues and suggestions tracked via GitHub
- All major decisions documented for reproducibility

---

*This roadmap is a living document and will be updated as the project evolves. The DataFrame-centric, variable-granularity approach is central to all phases and milestones.* 