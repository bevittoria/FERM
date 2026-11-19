Fixed truth, partially observed relation matrix. Version 1, 2026-10-16.
For each world, generate geography, populations and one iid Gaussian off-diagonal
relation matrix R, zero diagonal. No empirical covariates. The coefficient strength
is fixed; S*=strength*R and sigma=1. Population supplies opportunity counts.
Truth is an unsmoothed finite MC conditional law (abort zero acceptance origins).
Generate training and test once per world, reuse exactly for every mask/coverage.
Masking changes only the input, never S*, truth probabilities, or observations.
Per origin use independent random destination permutations, independent of R/flows.
Reveal the first floor(q*(n-1)) entries: nested masks, same count per row.
Unknown entries are zero shifts by modelling convention. No rescaling after masking.
The visible input equals R at known entries. Calibrate scalar theta on H1-like train
cross-entropy; theta=0 is analytic RM. This is not matrix completion: unseen
coefficients are not inferred. Known benefits can indirectly affect hidden-route
probabilities through shared threshold/search and conditional normalization.
Use independent truth/train/test/fit/check streams and common random numbers
across candidates/coverage within a mask for fit/check. Endpoints q=0/1 are shared
across masks, preventing artificial mask dependence at endpoints.
Fit/check candidate probabilities use 50 total RM pseudo-counts per origin.
Store all raw fit counts and selected check counts. Prior sensitivity 5/50/500
holds the selected theta fixed; it does not test calibration robustness.
Metrics: origin-equally weighted generating-law CPC and KL, independent test
cross-entropy, MAE of probabilities on known and hidden entries WITHOUT row
renormalization. Known/hidden supports change with coverage, so direct comparison
is descriptive. A fixed audit subset (last 20% per-row ranks) stays hidden through
75% coverage: report its MAE to assess changes on identical hidden corridors.
At full coverage the audit set becomes known and is excluded from that curve.
Aggregate mask results within world, then mean and 10–90% between worlds.
Masks are not independent worlds. No bootstrap, no quadrature, no p-values.
Main view: fixed true matrix alongside predictions at 0, one-third, two-thirds and full requested
coverage, fixed node ordering (x coordinate) and colour scales; signed errors
below. First world and first mask are fixed in advance, no best-case selection.
Second view: full allocation CPC, known/hidden MAE, and fixed-hidden audit MAE.
No guaranteed monotonic improvement, generalization or RM fallback claims.
New unique output directory and protocol saved before simulation. Pilot is a
numerical/implementation check, not a paper result. No automatic paper run.
