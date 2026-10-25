# nnInteractive CPU timing test

A research timing test: how long one click of the nnInteractive 3D segmentation model takes on an ordinary x86 computer with no graphics card, and how much memory it needs. It runs on GitHub Actions runners (Windows and Ubuntu) with model windows of 128³, 160³ and 192³ voxels. Research use only; not for clinical use.

- **What it does:** `timing.py` downloads one public CT scan and its reference outline, loads the model, places one click inside the tumour and two simulated corrections, and saves the timings, peak memory and overlap (Dice) to `results.json`.
- **Data (public, downloaded at run time from NCI Imaging Data Commons, not stored here):**
  - Zhao, B., Schwartz, L. H., Kris, M. G., & Riely, G. J. (2015). *Coffee-break lung CT collection with scan images reconstructed at multiple imaging parameters* (Version 3). The Cancer Imaging Archive. https://doi.org/10.7937/k9/tcia.2015.u1x8a5nr
  - Wee, L., Aerts, H. J. W. L., Kalendralis, P., & Dekker, A. (2020). *RIDER Lung CT Segmentation Labels from: Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2020.jit9grk8 (CC BY 3.0)
  - Fedorov, A., et al. (2023). National Cancer Institute Imaging Data Commons: Toward Transparency, Reproducibility, and Scalability in Imaging Artificial Intelligence. *RadioGraphics*, 43(12). https://doi.org/10.1148/rg.230180
- **Model:** nnInteractive v1.0. Isensee, F., Rokuss, M., Krämer, L., et al. (2025). *nnInteractive: Redefining 3D Promptable Segmentation.* arXiv:2503.08373. The weights are downloaded at run time from the official source and are licensed **CC BY-NC-SA 4.0** (non-commercial). They are not redistributed here.
