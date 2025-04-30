/*
📌 Purpose – Defines the phased, documentation-first roadmap for the de-novo branch of the Video Timeline Analyzer.
🔄 Latest Changes – Marked ingestion module as started; added next step recommendation for scene detection.
⚙️ Key Logic – Code development may now begin, following the modular folder structure and project rules.
📂 Expected File Path – docs/ROADMAP.md
🧠 Reasoning – Ensures a rigorous, reproducible, and maintainable foundation for future development, now transitioning to implementation, with a focus on solo, AI-assisted scientific coding.
*/

# Project Roadmap (De-Novo Branch)

## Overview

This roadmap defines the phases and milestones for the de-novo branch, focusing on documentation, design, project rule creation, and now codebase bootstrapping and initial development.

**Note:** This project is being developed by a single scientist-developer, with a focus on learning, fun, and leveraging AI/code assistants to minimize manual coding. The process is designed to be enjoyable and educational, not just productive.

---

## Phase 1: Documentation & Design

- [x] Review and refine system architecture ([ARCHITECTURE.md](ARCHITECTURE.md)), emphasizing the DataFrame-centric, variable-granularity approach.
- [x] Complete technical specifications ([SPECIFICATIONS.md](SPECIFICATIONS.md)), including data models and alignment logic.
- [x] Finalize development environment and setup documentation ([DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)).
- [x] Review and update all supporting documentation (e.g., [POWERSHELL_COMMANDS.md](POWERSHELL_COMMANDS.md), [DEVELOPMENT.md](DEVELOPMENT.md)).
- [x] Document all scientific and reproducibility requirements.

**Milestone:** All documentation reviewed and approved by project stakeholders.

---

## Phase 2: Project Rule Creation

- [x] Draft comprehensive Cursor project rules (see `.cursor/rules/cursorrules.mdc`).
- [x] Review and refine rules for:
    - Code structure and modularity
    - Documentation and commenting standards
    - Testing and validation requirements
    - Environment and dependency management
    - Git/GitHub workflow and commit policies
    - Scientific rigor and reproducibility
    - Data management and privacy
- [x] Approve and publish rules in the repository.

**Milestone:** Project rules finalized and published.

---

## Phase 3: Review & Planning for Implementation

- [x] Hold review meeting(s) to ensure all documentation and rules are clear, complete, and actionable.
- [x] Identify any gaps or ambiguities in the design or rules.
- [x] Plan the transition to implementation (future phases to be defined after review).

**Milestone:** Project is ready for codebase bootstrapping, with a clear, documented, and rule-driven foundation.

---

## Phase 4: Codebase Bootstrapping & Initial Development

- [x] Set up the modular folder structure (see `src/` and subfolders for each pipeline stage)
- [x] Begin implementation of core modules:
    - [x] Ingestion (video/audio loading) – started and tested
    - [ ] Scene detection
    - [ ] Audio analysis
    - [ ] Visual analysis
    - [ ] Metadata/DataFrame construction
    - [ ] Database (Qdrant) integration
    - [ ] (Optional) UI
- [ ] Write initial unit tests in `tests/`
- [ ] Ensure all code follows project rules for modularity, documentation, and reproducibility
- [ ] Commit and push all changes with clear, descriptive messages

**Milestone:** Core modules bootstrapped and under version control, ready for iterative development.

---

## Review & Feedback

- Regular review meetings at the end of each phase (self-review or with AI assistant).
- Issues and suggestions tracked via GitHub.
- All major decisions documented for reproducibility.

---

## Tips & References for Solo, AI-Assisted Scientific Development

- **Leverage AI/code assistants** (like Cursor, Copilot, or ChatGPT) for code generation, refactoring, and documentation.
- **Keep learning fun:** Try new tools, experiment, and don't be afraid to iterate or refactor.
- **Document your process:** Use README files, comments, and commit messages to track your learning and decisions.
- **Useful resources:**
    - [The Turing Way: Guide to Reproducible Research](https://the-turing-way.netlify.app/)
    - [Software Carpentry: Scientific Programming Best Practices](https://software-carpentry.org/lessons/)
    - [GitHub Guides: Mastering Markdown](https://guides.github.com/features/mastering-markdown/)
    - [Qdrant Documentation](https://qdrant.tech/documentation/)
    - [PySceneDetect Documentation](https://pyscenedetect.readthedocs.io/en/latest/)
    - [OpenAI Whisper](https://github.com/openai/whisper)
    - [CLIP and BLIP-2 Models](https://github.com/openai/CLIP), (https://github.com/salesforce/BLIP)

---

## Next Logical Step

**Implement and test the scene detection module.**
- Use the frames and metadata produced by the ingestion module.
- Integrate with PySceneDetect or your chosen scene detection tool.
- Save scene boundaries and prepare outputs for downstream audio and visual analysis.
- Document and version-control your progress.

*This roadmap is a living document and will be updated as the project evolves. The documentation- and rule-driven approach is central to all phases and milestones, now transitioning to implementation. Enjoy the journey!* 