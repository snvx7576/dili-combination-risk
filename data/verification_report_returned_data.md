# Verification Report: Returned PK and Metabolite Data
Checked by Cooper, 21 November 2026. Files received from Shivam (delegated research):
Hepatotoxicity_ALL_1336_PK_Parameters.csv (1,483 rows) and
Hepatotoxicity_ALL_1336_Active_Toxic_Metabolites.csv (1,469 rows).

## Coverage
- All 1,336 DILIrank drug names present in both files, exact match, none missing.
- PK sources: DailyMed 1,190 rows, FDA accessdata 67, UK medicines.org.uk 44,
  PubMed/PMC 61, others small. Metabolite sources: DailyMed 554, NCBI 499,
  PMC/PubMed 282, FDA 57.
- Engine-ready as returned (typical Tmax AND half-life present): 560 drugs.
- After Cooper's mechanical derivation (typical := midpoint of the label
  range, flagged per-cell in file 10; the same convention Layer 2 used):
  856 drugs. App library after oral-formulation selection and sanity
  filters: 785 drugs.
- Metabolites: 142 parent drugs with at least one TOXIC or BOTH metabolite;
  815 NONE_KNOWN rows; paracetamol row exactly right (NAPQI, TOXIC, more
  toxic than parent YES, CYP2E1).

## Spot checks against project-verified values (PK table v0.1)
- Diclofenac sodium delayed-release Tmax 2 h: MATCH. Extended-release
  Tmax 5.3 h vs verified 5.25: MATCH.
- Isoniazid Tmax 1.5, half-life 1.5 (range 0.5-5, acetylator-dependent):
  plausible and label-consistent.
- Amoxicillin Tmax 1.5, half-life 1.36; atorvastatin half-life 14: MATCH
  known label values.

## Issues found (recorded, not hidden)
1. Typical values often blank where the label gives only a range
   (537 cells). Resolved by flagged midpoint derivation, no invention.
2. Rabeprazole maximum doses in the returned file (single 100 mg, daily
   120 mg) conflict with the standard 20 mg label maximum. Corrected to
   20/20 in the app data via the project override; raw returned file kept
   unchanged for audit.
3. The returned aspirin row describes the parent compound in a 75 mg
   antiplatelet formulation (Tmax 0.33 h); the project model uses
   salicylate (Tmax 1.74 h, verified v0.1). Project value kept by override.
4. Extreme values that look like errors are genuine label values
   (amiodarone half-life 1,392 h; alendronate terminal half-life 87,600 h;
   bedaquiline 3,960 h; depot-injection Tmax values in days). Depot and
   non-oral rows are excluded from the app by the oral-formulation filter.
5. Diclofenac sodium max daily 225 mg in the returned file vs 150 mg in
   the project table: both appear on real labels (225 mg is the US
   rheumatoid-arthritis maximum). Project value kept by override.

## Precedence rule applied
Project-verified v0.1 fixed points override returned values for
acetaminophen, diclofenac sodium, diclofenac potassium, aspirin,
rabeprazole sodium and cyanocobalamin, so every published Layer 2
scenario number remains exactly reproducible in app v3 (verified:
scenario A Lmax 1.7356, AUC 11.0852).

## Status
Files loaded as Fixed Database 10 (PK v1) and 11 (metabolites v1).
Returned values beyond the spot checks above are label-sourced but not
yet individually re-verified; the app footer states this provenance.
