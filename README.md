# Spectral Transformation Lanczos for Symmetric-Definite GEP

This repository contains the implementation and research artifacts for the thesis:

**The Spectral Transformation Lanczos Algorithm for the Symmetric-Definite Generalized Eigenvalue Problem: A Comparative Analysis with Conditioning Insights**  
Ayobami Adebesin, Georgia State University (2025)  
Advisor: Michael Stewart

---

## 1) Research Overview

This project studies the generalized eigenvalue problem

$$
A v = \lambda B v
$$

for real symmetric matrices $A$ and $B$, with $B$ positive definite (and discussion extending to semidefinite settings in the thesis).

The central goal is to evaluate an **iterative** shifted-invert spectral method (ST-Lanczos) and compare its residual behavior against stability/error trends proven for **direct methods** in recent literature.

Key question investigated:

- Do the residual-bound patterns established for direct spectral-transformation methods also appear in an iterative Lanczos workflow?

---

## 2) Main Contributions

- Implementation of a **Spectral Transformation Lanczos (ST-Lanczos)** pipeline in Python.
- Controlled synthetic problem generation with known eigenvalue distributions.
- Residual analysis for:
  - Lanczos decomposition residual,
  - spectral-transformed Ritz residuals,
  - generalized residuals for mapped eigenpairs,
  - best-achievable residual proxies (via smallest singular values).
- Comparative discussion of decomposition choices for $A-\sigma B$:
  - `LU` factorization (implemented in the code path),
  - symmetric eigendecomposition (analyzed in thesis results).
- Empirical validation of shift-dependent residual behavior consistent with theoretical predictions.

---

## 3) Repository Structure

```text
.
|-- code/
|   |-- main.py
|   |-- spectral_lanczos.py
|   |-- plots/
|   |   |-- LU/
|   |   `-- WDW/
|   `-- *.mtx / generated figures
|-- thesis_research/
|   |-- main.tex
|   |-- Chapters/
|   |-- Plots/
|   `-- main.pdf
`-- presentation/
    |-- st_lanczos.tex
    `-- st_lanczos.pdf
```

### Important files

- `code/spectral_lanczos.py`: core numerical routines and residual computations.
- `code/main.py`: end-to-end experiment driver (matrix generation, ST-Lanczos run, residual plots).
- `thesis_research/main.pdf`: full thesis text (methodology, analysis, conclusions).
- `presentation/st_lanczos.pdf`: slide deck summary.

---

## 4) Method in Brief

1. Construct a controlled generalized eigenvalue problem $Av=\lambda Bv$:
   - Build a diagonal spectrum $D$,
   - form $C=QDQ^T$,
   - form SPD $B=L_0L_0^T+\delta I$,
   - set $A=LCL^T$ where $B=LL^T$.
2. Apply shifted-invert spectral transformation around shift $\sigma$.
3. Run Lanczos with full reorthogonalization on transformed operator.
4. Compute Ritz pairs and keep converged pairs by tolerance.
5. Map converged pairs back to generalized eigenpairs.
6. Analyze residual quality versus eigenvalue location and conditioning.

---

## 5) Key Experimental Findings (Thesis Summary)

- A **moderate shift** (not too close to eigenvalues) yields strong accuracy and stable residual behavior.
- Residuals are typically smallest for eigenvalues near/under the shift and degrade for eigenvalues much larger than the shift, consistent with predicted scaling trends.
- For `LU(A-\sigma B)` experiments:
  - decomposition residual observed around $10^{-11}$,
  - about $78\%$ Ritz convergence (reported setting in thesis),
  - generalized residuals near machine precision for favorable spectrum regions.
- Symmetry-preserving decompositions (e.g., eigendecomposition of $A-\sigma B$) exhibited stronger stability trends in reported experiments.

See `thesis_research/Chapters/Chapter_3.tex` and `thesis_research/main.pdf` for full derivations, plots, and caveats.

---

## 6) Reproducibility

### 6.1 Python environment

The thesis reports experiments with:

- Python 3.9.6
- NumPy 2.0.2
- SciPy 1.13.1
- Matplotlib

Install dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

### 6.2 Run the experiment code

From repository root:

```bash
cd code
python main.py
```

This script:

- generates synthetic matrices $A,B$,
- runs ST-Lanczos,
- prints conditioning/residual diagnostics,
- writes residual plots (e.g., `residual_lu_gs`).

> Note: default matrix sizes in `main.py` are large (3000x3000) and can require significant memory/runtime.  
> For quick checks, reduce `m1,m2,m3` and/or `n`.

---

## 7) Building the Thesis and Slides

Thesis source: `thesis_research/main.tex`  
Slides source: `presentation/st_lanczos.tex`

Typical LaTeX build sequence (if needed):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 8) Current Scope and Limitations

- The repository is primarily a **research artifact**, not a packaged library.
- The active code path emphasizes `LU`-based shifted solves; other decomposition variants discussed in the thesis are presented analytically/experimentally in the document.
- Parameters in code may differ from individual thesis experiment settings; consult the chapter text when reproducing a specific figure/table.

---

## 9) Suggested Future Extensions

- Add configurable experiment scripts/CLI for batch sweeps over shift and conditioning.
- Add sparse matrix pathways and benchmarking on larger structured problems.
- Integrate symmetry-preserving factorizations in code paths for direct comparison runs.
- Add automated experiment manifests and result logging for full reproducibility.

---

## 10) References (Core)

1. Stewart, M. (2024). *Spectral Transformation for the Dense Symmetric Semidefinite Generalized Eigenvalue Problem*. arXiv:2411.03534.  
2. Ericsson, T., & Ruhe, A. (1980). *The spectral transformation Lanczos method for the numerical solution of large sparse generalized symmetric eigenvalue problems*. Mathematics of Computation.  
3. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).  
4. Moler, C. B., & Stewart, G. W. (1973). *An Algorithm for Generalized Matrix Eigenvalue Problems*.

See `thesis_research/bibliography.bib` for the full bibliography used in the thesis.
