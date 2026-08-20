# Vision-Language Navigation for an Autonomous Mobile Robot: Reproduction, Environment Engineering, and Evaluation Planning of ETPNav

**Author:** Jiasu Hu
**Program:** Faculty Undergraduate Research Programme (FURP) 2025
**Supervisor:** Dr. Cui
**Track:** Vision-Language Navigation and RL Navigation for AMR — Research Track

---

## Abstract

This report documents an eight-week research effort aimed at reproducing **ETPNav** (Evolving Topological Planning for Vision-Language Navigation), a state-of-the-art method for Vision-Language Navigation in Continuous Environments (VLN-CE), as a candidate perception-and-planning module for an Autonomous Mobile Robot (AMR). The project followed the Research Track requirement of reproducing a cited paper's method on the Matterport3D-based R2R-CE benchmark. Substantial effort was spent on literature review of the VLN-CE task family, environment configuration of the Habitat-Sim / Habitat-Lab simulation stack, dependency resolution across conda environments, cloud deployment on a rented GPU instance, and acquisition of the Matterport3D 3D scene assets and pretrained checkpoints required by ETPNav. While the evaluation pipeline was brought to a near-runnable state — pretrained weights, evaluation checkpoint, R2R_VLNCE_v1-2 annotations, and connectivity graphs all verified and in place — full quantitative reproduction was not completed within the programme timeline due to a missing set of Matterport3D `.glb` scene mesh files, a known bottleneck in this reproduction pipeline. This report presents the task formulation, the ETPNav methodology, a full account of the engineering process (including a reusable troubleshooting log), a benchmark comparison drawn from the original literature to contextualize expected performance, and a concrete completion plan for the remaining evaluation step.

---

## 1. Introduction

### 1.1 Background and Motivation

Autonomous Mobile Robots (AMRs) are increasingly expected to operate in human-centric indoor environments where instructions are given in natural language rather than as coordinate waypoints (e.g., "go past the kitchen and turn left at the second door"). This capability, known as **Vision-Language Navigation (VLN)**, requires an agent to jointly interpret language instructions, perceive its visual surroundings, and produce a sequence of low-level actions that lead to a target location. Unlike navigation in a discretized graph of pre-defined viewpoints, real AMRs must operate in **continuous** state and action spaces, which introduces additional challenges: accumulating pose error, collision avoidance, and the need to plan over long, partially unknown trajectories.

This motivates the study of **VLN-CE (VLN in Continuous Environments)**, introduced by Krantz et al. (2020), which ports the original discrete Room-to-Room (R2R) benchmark into the Habitat simulator, replacing graph traversal with continuous low-level control. This project focuses on reproducing **ETPNav**, one of the strongest published methods on VLN-CE, as a first step toward integrating a language-conditioned high-level planner into the group's broader AMR navigation stack (which also explores reinforcement-learning-based low-level navigation, per the supervisor's project scope).

### 1.2 Research Track Requirements

Per the project charter, the Research Track requires: (1) reproduction of at least one method from a cited paper, (2) a minimum 10% methodological or empirical extension beyond the original work, and (3) weekly documentation of environment setup, commands, configuration, and errors as research evidence. This report is structured around those deliverables, with an honest account of which have been fully met and which remain in progress.

### 1.3 Report Structure

Section 2 reviews related work and situates ETPNav within the evolution of VLN-CE methods. Section 3 formalizes the task. Section 4 details the ETPNav methodology and the reproduction plan. Section 5 presents the weekly progress log. Section 6 documents the engineering/reproducibility case study. Section 7 discusses the evaluation protocol and benchmark context. Section 8 discusses limitations and lessons learned. Section 9 concludes and outlines future work.

---

## 2. Related Work

### 2.1 From Discrete to Continuous VLN

The original **Room-to-Room (R2R)** benchmark (Anderson et al., 2018) formulated VLN as navigation over a discrete graph of panoramic viewpoints, where an agent "teleports" between nodes. While this enabled early progress, it abstracted away the low-level control problem central to real robot deployment. **VLN-CE** (Krantz et al., 2020) addressed this by porting R2R instructions into the Habitat simulator over continuous Matterport3D meshes, requiring agents to output low-level actions (`move_forward`, `turn_left`, `turn_right`, `stop`) and cope with realistic collision and pose drift. Their baseline, a **Cross-Modal Attention (CMA)** sequence-to-sequence agent, reported a Success Rate (SR) of only 0.29 and SPL of 0.27 on the Val Unseen split — underscoring the difficulty of the continuous setting relative to discrete R2R, where comparable models exceeded 0.55 SR.

### 2.2 Waypoint Prediction and Topological Memory

A major line of follow-up work sought to bridge discrete and continuous VLN by predicting candidate **waypoints** directly from panoramic RGB-D observations (Hong et al., 2022, "Bridging the Gap Between Learning in Discrete and Continuous Environments"), allowing discrete-style graph-based policies to be executed in continuous space. In parallel, methods such as **DUET** (Chen et al., 2022) introduced dual-scale graph transformers that maintain a coarse topological map of visited and predicted-but-unvisited nodes, enabling more effective long-horizon planning and backtracking than purely recurrent agents.

### 2.3 ETPNav: Evolving Topological Planning

**ETPNav** (An et al., 2024, *IEEE TPAMI*) unifies these directions for the continuous setting. It removes the assumption of a pre-built navigation graph and instead constructs an **online, evolving topological map** as the agent explores, self-organizing predicted waypoints into graph nodes and edges in real time. A **cross-modal graph planner**, built on a Transformer architecture, then reasons jointly over this evolving graph and the language instruction to select the next macro-action (i.e., which topological node to head toward), while a lower-level controller executes local point-goal navigation with obstacle avoidance between nodes. The authors report SR = 0.572 and SPL = 0.492 on R2R-CE Val Unseen, and describe improvements of over 10% (R2R-CE) and over 20% (RxR-CE) relative to prior state-of-the-art at the time of publication. This combination of long-horizon topological reasoning with continuous-environment robustness is the primary reason ETPNav was selected as the reproduction target for this project, and why it is a plausible high-level planning module for an AMR that must also handle continuous, dynamic real-world spaces.

### 2.4 Positioning of This Project Relative to RL-Based Navigation

The supervisor's broader research group also investigates **reinforcement-learning (RL) based navigation**, which typically learns a reactive or semi-reactive control policy directly from reward signals (e.g., distance-to-goal, collision penalty) without explicit language grounding. VLN-based methods like ETPNav, by contrast, are trained with supervised imitation and auxiliary losses over language-annotated trajectories, and excel at long-horizon, instruction-following tasks but are typically more data-hungry and simulator-dependent. A natural direction for the AMR platform is a **hierarchical combination**: an ETPNav-like module handling high-level, language-conditioned waypoint selection, with an RL-trained local controller handling low-level obstacle avoidance and dynamics — mirroring ETPNav's own two-level design internally, but potentially replacing its local controller with a learned RL policy tailored to the target robot's dynamics. This is discussed further as future work in Section 9.

---

## 3. Problem Formulation

Following Krantz et al. (2020), the VLN-CE task is defined as follows. At each episode, the agent receives a natural language instruction $I = (w_1, w_2, \dots, w_L)$ describing a path through an indoor environment reconstructed from Matterport3D scans. At each timestep $t$, the agent receives an RGB-D panoramic (or limited field-of-view) observation $o_t$ and must select a low-level action $a_t \in \{\text{forward}, \text{turn-left}, \text{turn-right}, \text{stop}\}$. An episode succeeds if the agent issues `stop` within a threshold distance $d_{th}$ (typically 3m) of the goal location within a maximum step budget.

Standard evaluation metrics are:

- **NE (Navigation Error):** the geodesic distance, in meters, between the agent's final position and the goal.
- **SR (Success Rate):** the fraction of episodes with NE less than $d_{th}$.
- **OSR (Oracle Success Rate):** success is counted if the closest point along the *entire trajectory* (not just the final position) satisfies NE less than $d_{th}$.
- **SPL (Success weighted by Path Length):** SPL jointly penalizes both failure and inefficient (overly long) successful trajectories, and is generally treated as the primary ranking metric in VLN-CE literature.

This project targets the **R2R-CE Val Unseen** split, which evaluates generalization to environments not seen during training, using ETPNav's officially released pretrained checkpoint.

---

## 4. Methodology

### 4.1 ETPNav Architecture

ETPNav operates in two coupled stages:

1. **Evolving Topological Mapping.** At each step, a waypoint predictor (trained following Hong et al., 2022) proposes candidate navigable directions/distances from the current panoramic observation. These candidates are incrementally merged into a persistent topological graph $G = (V, E)$, where nodes represent visited or predicted-reachable locations and edges represent local navigability. Because the graph is built online rather than assumed to be known in advance, ETPNav does not require privileged access to the environment's navmesh at inference time, making it more representative of what a physically deployed AMR would have access to.

2. **Cross-Modal Graph Planning.** A Transformer-based planner encodes the language instruction and the current graph state jointly (via cross-attention between instruction tokens and node/edge features), and outputs a probability distribution over graph nodes — including both nodes already visited (enabling backtracking) and frontier nodes (enabling exploration). The selected target node is then reached via a local point-goal controller with obstacle avoidance, decomposing the continuous low-level control problem away from the high-level language-grounded reasoning problem.

### 4.2 Reproduction Plan

The intended reproduction pipeline, following the official ETPNav repository, consists of:

1. Installing the pinned simulation stack: **Habitat-Sim 0.1.7** and **Habitat-Lab 0.1.7** (ETPNav is not compatible with the newer Habitat 2.x/3.x API, which restructures the sensor and action space interfaces).
2. Acquiring the **Matterport3D** dataset — specifically the `.glb` textured mesh files for the ~90 building-scale scenes used by R2R-CE — which requires signing and submitting the official MP3D Terms of Use to obtain a download script from the dataset authors.
3. Acquiring the **R2R_VLNCE_v1-2** instruction annotations and **connectivity graph** (`connectivity_graphs.pkl`) files released alongside VLN-CE.
4. Downloading ETPNav's released **pretrained representation weights** (`mlm.sap_r2r`, an initialization pretrained with masked-language-modeling and single-step action prediction objectives) and the **fine-tuned evaluation checkpoint** (`ckpt.iter12000.pth`).
5. Running the official evaluation script against the Val Unseen split to compute NE / SR / OSR / SPL.

Each of these five steps is mapped against actual progress in Sections 5–6 below.

---

## 5. Weekly Progress Log

The following table summarizes the eight weeks of work, condensed from the full research log maintained at `docs/00_weekly.md` in the project repository.

| Week | Key Activities | Outputs / Status |
|---|---|---|
| 1 | Repository and documentation structure set up; literature review of VLN-CE task family and core obstacles (sim-to-real gap, continuous action space, dataset licensing) | Reading notes; project scaffolding |
| 2 | Installed Habitat-Sim locally; resolved headless/offscreen rendering issues (OSMesa configuration); ran the official smoke test, successfully initializing an agent in the `apartment_1.glb` demo scene | Verified local Habitat-Sim installation |
| 3 | Selected **ETPNav** as the reproduction target and baseline; applied for official Matterport3D dataset access (application not approved within the timeline); obtained an alternative copy of the MP3D/R2R data via a community-shared source; provisioned a rented AutoDL cloud GPU instance and began transferring pretrained weights and R2R data | Baseline selected; cloud instance provisioned; dataset partially acquired |
| 4 | Extensive debugging of the cloud-side Habitat-Sim installation, including conda "solving environment" deadlocks; iterated through dependency version combinations | First successful scene load and agent instantiation on the cloud instance |
| 5 | Diagnosed that using the latest available versions of core dependencies (habitat-sim, habitat-lab, torch) caused cascading version conflicts; pinned to the older versions specified by the ETPNav authors, which resolved most conflicts; discovered the previously downloaded pretrained model file was incomplete/corrupted; freed cloud disk space (constrained to ~30GB) and re-downloaded | Stable dependency set (`etpnav` conda env); verified pretrained weight integrity |
| 6 | Confirmed presence of: conda environment (`etpnav`), pretrained representation weights (`mlm.sap_r2r.pt`), evaluation checkpoint (`ckpt.iter12000.pth`), `R2R_VLNCE_v1-2` annotations, `connectivity_graphs.pkl`; ran the evaluation script, which failed during environment initialization with a missing-scene-file error | Evaluation pipeline blocked on missing Matterport3D `.glb` scene meshes |
| 7 | Investigated the scene-file gap; confirmed via the official ETPNav repository documentation that the full 90-scene MP3D mesh set (distinct from the annotation/connectivity files already in place) must be separately obtained through the MP3D Terms of Use process; re-submitted the official dataset request and searched for supplementary community mirrors for the remaining scenes | Root cause of blockage identified and documented; second data request in progress |
| 8 | Consolidated the full engineering log into a reproducibility case study; drafted evaluation protocol and completion plan; compiled this report | Final report and troubleshooting appendix |

---

## 6. Reproducibility Case Study: Deploying ETPNav on Habitat-Sim

This section is intended as a standalone, reusable reference for the specific engineering obstacles encountered, since they consumed the majority of project time and are, in the author's experience, under-documented relative to their frequency in VLN-CE reproduction attempts.

**6.1 Dataset access friction.** The Matterport3D dataset is not freely downloadable; it requires submitting an academic Terms of Use agreement to the original dataset authors, and approval turnaround is not guaranteed within a short research window. This is a structural bottleneck for any time-boxed VLN-CE reproduction project and should be budgeted for at the very start of a project, not after baseline selection.

**6.2 Dependency version sensitivity.** ETPNav pins Habitat-Sim/Habitat-Lab to version 0.1.7, which predates the significant Habitat 2.x API redesign. Attempting to install against a more recent Habitat version — the default outcome of a naive `pip install` or `conda install` without explicit version pinning — leads to import errors and silent API mismatches (e.g., in sensor specification and action space registration) rather than clear installation failures, making root-causing time-consuming. Explicit version pinning to the exact commit/tag used in the original codebase is essential before any other debugging is attempted.

**6.3 Headless rendering configuration.** Habitat-Sim's default rendering backend assumes a display context. On a cloud GPU instance without an attached display, EGL/OSMesa-based offscreen rendering must be explicitly configured; failures here typically manifest as segmentation faults or driver-context errors that do not obviously point to a rendering misconfiguration.

**6.4 Storage constraints.** The rented cloud instance's default disk allocation (~30GB) is insufficient for the combined footprint of the MP3D scene set, R2R-CE annotations, and multiple pretrained checkpoints. Incomplete downloads caused by disk-full conditions can silently produce a truncated but loadable checkpoint file, leading to confusing downstream errors (e.g., shape mismatches during model loading) that appear unrelated to the actual root cause. Verifying file checksums/sizes immediately after every large download is recommended practice going forward.

**6.5 Annotation files vs. scene mesh files are separate deliverables.** A specific and easily-missed pitfall: the R2R_VLNCE annotation JSONs and the `connectivity_graphs.pkl` file (which encodes navigable viewpoint graphs) can be obtained and verified as "complete" while the actual 3D `.glb` mesh files for the underlying buildings are still missing. Because both are nominally "the dataset," it is easy to assume readiness prematurely. This was precisely the failure mode encountered in Week 6–7 of this project.

---

## 7. Evaluation Protocol and Benchmark Context

### 7.1 Planned Evaluation Protocol

Once the full Matterport3D scene set is obtained, the evaluation plan is as follows:

1. Run `habitat-lab`'s scene-loading sanity check against all 11 Val Unseen scenes to confirm mesh integrity before invoking the full ETPNav evaluation script.
2. Execute ETPNav's evaluation entry point with the released `ckpt.iter12000.pth` checkpoint against the `R2R_VLNCE_v1-2` Val Unseen split, computing NE, SR, OSR, and SPL as defined in Section 3.
3. Log per-episode trajectories for qualitative inspection (in particular, failure cases where the agent stops far from the goal, versus cases where it reaches the goal region but fails to issue `stop`).
4. Compare obtained metrics against the officially reported numbers (Table 1 below) to sanity-check the reproduction's fidelity, accounting for expected minor variance due to simulator version and stochastic seeding differences.

### 7.2 Reference Benchmark Numbers

The following table reports figures **as published in the original literature**, included here to contextualize the expected performance range of the reproduction target once evaluation is unblocked. These are not results produced by this project.

**Table 1: Published R2R-CE Val Unseen results (for reference only)**

| Method | Source | NE ↓ (m) | OSR ↑ | SR ↑ | SPL ↑ |
|---|---|---|---|---|---|
| Seq2Seq (CMA baseline) | Krantz et al., 2020 | 7.60 | 0.36 | 0.29 | 0.27 |
| Waypoint Predictor + CMA | Hong et al., 2022 | 6.31 | 0.52 | 0.43 | 0.36 |
| ETPNav | An et al., 2024 | — | 0.66 | 0.572 | 0.492 |

*(NE for ETPNav not restated here as it was not directly reported in the same units in the reviewed source; figure omitted rather than approximated.)*

This table serves two purposes in this report: it substantiates why ETPNav was selected as the reproduction target (largest reported SPL margin over the original baseline among reviewed methods), and it defines the numerical range against which this project's eventual reproduction run — once the scene-file blocker is resolved — should be evaluated for correctness.

---

## 8. Discussion and Limitations

The central limitation of this project is that **no quantitative VLN-CE evaluation results were produced** within the programme timeline; the pipeline reached an initialization-time failure rather than completing an evaluation run. This should be stated plainly rather than obscured. That said, the project did substantively satisfy the Research Track's evidentiary requirements around environment reproduction: the specific baseline paper (ETPNav) and task (R2R-CE Val Unseen) were correctly identified and scoped; the non-trivial simulation and dependency stack was correctly diagnosed and largely resolved (Weeks 2–5); and four of the five required reproduction assets (conda environment, pretrained weights, evaluation checkpoint, annotation/connectivity files) were fully verified, with the fifth (MP3D scene meshes) identified as blocked on an external, non-technical dependency (dataset licensing approval) rather than a technical failure on the author's part.

A secondary limitation is that the "10% innovation" requirement of the Research Track has not yet been addressed, since it presupposes a working reproduction as its foundation. The most natural candidate extension, given the project's trajectory, is the hierarchical VLN+RL integration sketched in Section 2.4 — replacing ETPNav's local point-goal controller with an RL-trained controller matched to the target AMR's specific kinematics and sensor suite — which becomes feasible to prototype once baseline evaluation is unblocked.

---

## 8.5 Proposed Directions for Improving SPL and SR (Conceptual, Untested)

Although no experiments were run to validate the following ideas, they are proposed here as concrete, literature-grounded directions motivated by (a) a summer of reading the VLN-CE literature, and (b) the specific failure modes observed while trying to get ETPNav itself running. Each proposal states its mechanism, the reasoning that motivates it, and how it would be evaluated if implemented. None of these have been empirically validated by this project; they are candidate hypotheses for future work, not claimed results.

**8.5.1 Uncertainty-gated stopping to reduce false-positive early termination.**
A meaningful share of SPL loss in topological-graph agents like ETPNav plausibly comes not from failing to reach the goal region at all, but from two more subtle failure modes: (a) stopping at a node that is spatially near, but not actually at, the target, flipping a near-success into a failure by pushing NE just past the 3m threshold; and (b) overshooting or wandering near the goal before stopping, which lengthens the trajectory and depresses SPL even on trials that nominally succeed. ETPNav's stop decision is a single softmax head over graph nodes with no explicit confidence calibration, so both failure modes are structurally invisible to the model itself — it has no mechanism to "know that it doesn't know." My proposal is to attach a lightweight uncertainty estimator on top of the existing action logits, implemented either as MC-dropout at inference time or as a small 3-way ensemble sharing the frozen backbone, that estimates the planner's confidence in its current stop decision at each step. When estimated confidence falls below a calibrated threshold, the agent is required to take one additional short verification action (e.g., a small forward step or a panoramic re-scan) before committing to `stop`, rather than stopping immediately on the first candidate node that crosses the softmax argmax. This is motivated directly by the SR–OSR gap already visible in the published numbers in Section 7.2 (e.g., ETPNav's OSR of 0.66 versus SR of 0.572): a non-trivial fraction of episodes pass near the goal at some point in the trajectory (oracle success) but the agent's *actual* stopping decision fails to capture it, which is exactly the pattern an uncertainty-gated stopping rule targets. Evaluation would compare SR/SPL with and without the verification step on the same checkpoint, and separately report how often the verification step was triggered near true goal locations versus false positives, to check whether the mechanism is actually catching the intended failure mode rather than just adding noise.

**8.5.2 Frontier-aware exploration bonus to reduce wasted backtracking.**
ETPNav's evolving topological map allows the planner to select among both visited and frontier (unvisited-but-predicted) nodes at every step, but the planner is trained purely to imitate expert trajectories via supervised cross-entropy, with no explicit incentive to prefer efficient exploration over redundant revisiting. In long, ambiguous instructions this can plausibly manifest as the agent oscillating between a small set of already-visited nodes rather than committing to unexplored frontier regions — a behavior that directly inflates trajectory length $\ell_i$ in the SPL formula (Section 3) without necessarily causing outright failure. Drawing on the exploration-bonus literature from RL-based navigation (where novelty/frontier bonuses are a standard technique for encouraging efficient coverage), my proposal is to add a small auxiliary loss term during fine-tuning that penalizes the planner's node-selection distribution for repeatedly re-selecting already-visited nodes when unexplored frontier nodes with instruction-relevant features (see 8.5.3 below) remain available, effectively regularizing the model toward monotonic-ish forward progress except when backtracking is clearly justified (e.g., near the end of the trajectory when the goal region has already been observed). This would be evaluated primarily via the mean trajectory-length ratio $\ell_i / \ell_i^*$ across successful episodes — a direct, interpretable proxy for the SPL efficiency penalty — rather than SR alone, since the mechanism is explicitly targeted at efficiency rather than raw success.

**8.5.3 Explicit landmark grounding for unseen-environment generalization.**
Natural-language R2R-style instructions typically contain concrete, spatially-anchored noun phrases ("the kitchen," "the staircase," "the painting on the wall") that are locally very informative but are used by ETPNav only indirectly, folded into a single instruction encoding that is reused identically at every planning step regardless of what is currently visible. My proposal is to explicitly extract noun-phrase landmarks from the instruction using the language encoder already present in the pipeline, and compute a CLIP-style visual-semantic similarity between each frontier node's most recent panoramic observation and the extracted landmark phrases, using this similarity as an additional prior that re-ranks frontier candidates before the cross-modal planner commits to its final choice. This is motivated by a recurring theme in recent VLN-CE work on integrating open-vocabulary semantic priors into navigation policies, where such grounding has been reported to improve generalization to unseen environments specifically because it reduces reliance on memorized layouts in favor of explicit, transferable landmark cues — a property that matters directly for the Val *Unseen* split emphasized throughout this report, where the agent cannot rely on having seen the specific building layout before. Evaluation would test whether SR/SPL gains, if any, are concentrated in trajectories with explicit, extractable landmark phrases versus more abstract instructions lacking them, which would help confirm or falsify the proposed mechanism rather than simply reporting an aggregate improvement.

**8.5.4 A hybrid VLN+RL local controller matched to the target AMR (the extension proposed in Section 2.4, made concrete).**
Section 2.4 already sketched replacing ETPNav's local point-goal controller with an RL-trained one; here the point is to be concrete about *why* this specifically should help SPL, rather than treating it as a generic integration exercise. ETPNav's local controller is trained and tuned against Habitat's idealized kinematic agent model, which does not reflect the acceleration limits, turning radius, or collision geometry of any specific physical AMR platform. Any mismatch here would manifest as extra corrective steps near obstacles or doorways — precisely the kind of small, repeated inefficiency that erodes SPL without necessarily causing outright navigation failure, and one that would only become visible once the agent is evaluated with AMR-realistic kinematics rather than Habitat's default agent. My proposal is to keep ETPNav's high-level topological planner frozen — since it is the component actually validated against the R2R-CE benchmark, and retraining it is out of scope for a platform-specific fix — and train only the local controller with RL inside a simulated model of the target AMR's actual kinematics and sensor-noise profile, using the high-level planner's selected node as the RL agent's per-step goal, with a reward shaped by progress toward that node and a collision penalty. This is a standard hierarchical RL pattern in general, but applying it specifically to close the sim-to-real *kinematic* gap for VLN's local execution step — as opposed to using RL as a full end-to-end alternative to VLN, which is more common in the literature — is, to my reading, an under-explored middle ground, and is also the most natural bridge between this project's VLN track and the group's parallel RL-navigation track. Evaluation would compare SPL and per-episode collision counts between the default Habitat-kinematics controller and the AMR-matched RL controller inside a kinematics-faithful simulation, prior to any real hardware trial.

**A note on epistemic status.** All four proposals above are reasoned from the structure of the evaluation metrics (Section 3), the documented SR–OSR gap and published benchmark numbers (Section 7.2), and general patterns observed across the VLN-CE literature reviewed for this report — not from any experiments run within this project. They are presented as testable hypotheses with a stated mechanism and evaluation plan, not as claimed improvements, and are offered as a starting point for the next phase of this work.

---

---

## 9. Conclusion and Future Work

This project undertook the reproduction of ETPNav, a topological-map-based Vision-Language Navigation method, as a candidate high-level planning module for an AMR. Across eight weeks, the work covered task and literature review, local and cloud-based simulation environment setup, systematic resolution of a series of dependency and infrastructure failures, and partial acquisition of the required datasets and pretrained models. The evaluation pipeline is verified complete up to scene-mesh loading, currently blocked by outstanding Matterport3D dataset access.

Immediate next steps are: (1) finalize Matterport3D scene access (official approval or a verified complete community mirror) and complete the first full R2R-CE Val Unseen evaluation run; (2) compare obtained NE/SR/OSR/SPL against the published reference numbers in Table 1 to validate reproduction fidelity; (3) begin prototyping the hierarchical VLN+RL controller described in Section 8 as the project's substantive extension beyond pure reproduction, in coordination with the group's parallel RL-navigation track.

---

## References

1. Anderson, P. et al. (2018). *Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments.* CVPR.
2. Krantz, J., Wijmans, E., Majumdar, A., Batra, D., & Lee, S. (2020). *Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments.* ECCV.
3. Hong, Y. et al. (2022). *Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation.* CVPR.
4. Chen, S. et al. (2022). *Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation (DUET).* CVPR.
5. An, D., Qi, Y., Li, Y., Huang, Y., Wang, L., Tan, T., & Shao, J. (2024). *ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
6. Chang, A. et al. (2017). *Matterport3D: Learning from RGB-D Data in Indoor Environments.* 3DV.

---

## Appendix A: Environment Specification

- Simulator: Habitat-Sim 0.1.7 / Habitat-Lab 0.1.7
- Python environment: `conda env etpnav`
- Compute: AutoDL rented cloud GPU instance
- Rendering backend: OSMesa (headless/offscreen)
- Pretrained assets: `mlm.sap_r2r.pt` (representation pretraining), `ckpt.iter12000.pth` (fine-tuned evaluation checkpoint)
- Datasets: `R2R_VLNCE_v1-2` annotations, `connectivity_graphs.pkl`, Matterport3D scene meshes (pending)

## Appendix B: Full Weekly Log

The complete, unabridged weekly research log — including exact commands, error tracebacks, and configuration files referenced in Section 6 — is maintained at:
`docs/00_weekly.md` in the project repository:
https://github.com/hujiasu/FURP-2025-JIasu-Hu-Vision-Language-Navigation-for-an-AMR
