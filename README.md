# Multi-Drug Combination Hepatotoxicity Risk Prediction

A free, public research tool that estimates the combined liver stress of medicines taken together, and shows when taking them at different times makes the combination safer.

**Project by Shivam Kumar**

Shivam Kumar designed this framework, directed the research, and takes responsibility for the project: its design decisions, its data verification, and everything published here.

**Live demo:** https://dili-combination-risk-psfzemxkcryjus4o39cmuh.streamlit.app/

## What this is

Most drug safety information describes one medicine at a time. Real patients, especially older patients and people treated for tuberculosis, HIV, or cancer, often take several medicines together. This project asks a simple question: can public data, molecular structures, and the timing of doses be combined to estimate the liver stress of a whole combination, beyond the score of any single drug in it?

The demo is built verdict first, for people rather than for specialists:

1. Enter your medicines, the dose in mg, and how many times a day each one is taken.
2. Read one clear verdict: LOW, MODERATE, or HIGH liver stress, in plain language.
3. See your day as a timeline: which hours are danger hours, which hours are raised stress, and which hours are safe.
4. Get one practical suggestion, computed by the model: how many hours later to take one of the medicines so that the peaks stop overlapping, with the new peak stated honestly, including when the danger cannot be removed completely.
5. Notes stay in a small collapsed section, and the full graph, numbers, and per-drug detail sit one tap away under "See the science".

The library behind the demo covers 785 drugs with verified pharmacokinetic values, drawn from a fixed database of all 1,336 drugs in DILIrank 2.0.

## Key results

| Finding | Value |
|---|---|
| Staggering the same doses by 6 hours cut the peak combined load | 26.9% (1.736 down to 1.269), at near-identical total exposure |
| Structural alerts on their own | AUROC 0.54, essentially chance: alerts cannot separate dangerous from safe drugs |
| Real-world combination reports assembled from FAERS 2026 Q3 | 4,368 reports with 2 to 4 drugs and a liver reaction |
| Drugs in the live app library | 785, with label-sourced timing values |
| Drugs with a known toxic metabolite | 142 |

## How it works

The framework has four layers. The bottom layer never changes; each layer above it reads from the ones below.

### Layer 0: Fixed Truth Database

A read-only database assembled once, from public sources only, and then frozen: DILIrank 2.0 liver-injury labels for 1,336 drugs, FAERS spontaneous-report data, a structure table (1,230 of 1,336 drugs matched to a molecular structure, 92.1%), a library of 13 structural alerts, a pharmacokinetic table, and a metabolite list. Models read this database; they never edit it. The data files in this repository, with their provenance, are in the `data/` folder.

### Layer 1: Structural alerts

Each drug structure is matched against 13 known hepatotoxic substructures (toxicophores) using RDKit SMARTS patterns. A match adds a note and points. It never produces a verdict on its own, and the results below show why: 560 of 1,224 matched drugs carry at least one alert, but the alert rate is nearly identical in the most concerning and the least concerning label groups (48.8% versus 45.6%).

### Layer 2: Temporal hepatotoxic load

Hepatotoxicity is a hazard; liver injury is the event that can happen on top of it. Each drug is given a concentration-time curve from its labelled timing values (Tmax and half-life), using a one-compartment Bateman model, and weighted by its hazard:

- Hazard H = (SeverityClass / 8) x concern weight, where the concern weight is 1.0 for vMost, 0.6 for vLess, 0.45 for Ambiguous, and 0.25 for vNo DILIrank concern.
- Load of one drug at time t: H x dose fraction x curve value at t. The dose fraction is the entered dose divided by the labelled reference maximum single dose.
- Combined load L(t) is the sum across all drugs and all doses in 24 hours.
- Outputs: peak load (Lmax), total exposure (toxic AUC), hours above the danger threshold of 0.75, and the danger window.

If two drugs peak in the same hours, their loads add and can cross the threshold. The same drugs, staggered, score lower. A zero-hazard control (vitamin B12) is built into the validation: adding it changes nothing, exactly as it should.

Model v2.1 adds the metabolite adjustment: where a drug forms a toxic metabolite, its hazard is multiplied by 1.25 if that metabolite is more toxic than the parent drug, or by 1.10 if the metabolite is toxic but not more toxic than the parent, with the adjusted hazard capped at 1.0.

### Layer 3: Learning layer

Machine learning models were trained on the same fixed data, on a held-out protocol (stratified 5-fold cross-validation, 562 drugs labelled vMost versus vNo):

- Structural alerts alone: AUROC 0.54, at chance level.
- Random forest on molecular fingerprints plus alerts: AUROC 0.813, AUPRC 0.697, MCC 0.429. This is the benchmark.
- Graph neural networks (GCN, GAT, and a GCN with alert fusion): best AUROC 0.748. The graph models fell short of the fingerprint benchmark, and that negative result is reported as it stands, with the next directions named: hyperparameter search, pretraining, and scaffold-based splits.

## Data

Every dataset used is public. Please cite the original sources, listed here, when reusing this work.

| Database | What it provides here | Citation and link |
|---|---|---|
| DILIrank 2.0 (FDA Liver Toxicity Knowledge Base, LTKB) | Liver-injury concern labels and severity classes for 1,336 drugs | Chen M, et al. (2016); Olubamiwa et al. (2025). https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset |
| FAERS (FDA Adverse Event Reporting System) | Real-world reports of medicines actually taken together, 2026 Q3 quarter used here | U.S. Food and Drug Administration. https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html |
| PubChem | Molecular structures (SMILES, InChIKey) for the structure table | Kim S, et al. (2023), Nucleic Acids Research. https://pubchem.ncbi.nlm.nih.gov |
| ChEMBL | Backup structure source, exact-match recoveries only | Zdrazil B, et al. (2024), Nucleic Acids Research. https://www.ebi.ac.uk/chembl/ |
| Tox21, via MoleculeNet | External validation set of 7,831 compounds with toxicity assay labels | Wu Z, et al. (2018), Chemical Science (MoleculeNet). Tox21 is a U.S. federal research collaboration. |
| LiverTox | Clinical reference for drug-induced liver injury, used for interpretation and cross-checking | LiverTox: Clinical and Research Information on Drug-Induced Liver Injury, NCBI Bookshelf, National Library of Medicine. |

The processed data files shipped in this repository live in the `data/` folder. Each file's origin, build date, and processing steps are documented in `data/DATA_SOURCES.md`, and the independent verification of the pharmacokinetic and metabolite data is in `data/verification_report_returned_data.md`.

## Repository contents

- `app.py`: the Streamlit demo (verdict-first interface, day timeline, computed staggering advice).
- `drugs_data.json`: the 785-drug app library, built from the fixed database.
- `requirements.txt`: Python dependencies (streamlit, pandas, plotly, numpy).
- `data/`: the fixed-database files listed above, with provenance.

## References

1. Chen M, Suzuki A, Thakkar S, Yu K, Hu C, Tong W. DILIrank: the largest reference drug list ranked by the risk for developing drug-induced liver injury in humans. Drug Discovery Today, 2016.
2. Olubamiwa et al. DILIrank 2.0 dataset release, FDA Liver Toxicity Knowledge Base, 2025.
3. Kim S, et al. PubChem 2023 update. Nucleic Acids Research, 2023.
4. Zdrazil B, et al. The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Research, 2024.
5. Wu Z, et al. MoleculeNet: a benchmark for molecular machine learning. Chemical Science, 2018.
6. U.S. Food and Drug Administration. FDA Adverse Event Reporting System (FAERS), quarterly data extracts.
7. LiverTox: Clinical and Research Information on Drug-Induced Liver Injury. NCBI Bookshelf, National Library of Medicine.

## Disclaimer

This is a research prototype only. It is not for clinical decision-making. Scores are model estimates built from public label and reporting data; they are associative, not causal, and they are not a prediction of any individual patient's risk. Nothing here replaces the advice of a doctor or pharmacist. Do not change how you take any medicine based on this tool without speaking to a doctor or pharmacist first.

## License

[License to be added by the author]
