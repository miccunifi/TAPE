# TAPE (WACV 2024)

### Reference-based Restoration of Digitized Analog Videotapes

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.14926)
[![GitHub Stars](https://img.shields.io/github/stars/miccunifi/TAPE?style=social)](https://github.com/miccunifi/TAPE)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reference-based-restoration-of-digitized/analog-video-restoration-on-tape)](https://paperswithcode.com/sota/analog-video-restoration-on-tape?p=reference-based-restoration-of-digitized)

This is the **official repository** of the [**paper**](https://arxiv.org/abs/2310.14926) "*Reference-based Restoration of Digitized Analog Videotapes*".

## Overview

### Abstract

Analog magnetic tapes have been the main video data storage device for several decades. Videos stored on analog videotapes exhibit unique degradation patterns caused by tape aging and reader device malfunctioning that are different from those observed in film and digital video restoration tasks. In this work, we present a reference-based approach for the resToration of digitized Analog videotaPEs (TAPE). We leverage CLIP for zero-shot artifact detection to identify the cleanest frames of each video through textual prompts describing different artifacts. Then, we select the clean frames most similar to the input ones and employ them as references. We design a transformer-based Swin-UNet network that exploits both neighboring and reference frames via our Multi-Reference Spatial Feature Fusion (MRSFF) blocks. MRSFF blocks rely on cross-attention and attention pooling to take advantage of the most useful parts of each reference frame. To address the absence of ground truth in real-world videos, we create a synthetic dataset of videos exhibiting artifacts that closely resemble those commonly found in analog videotapes. Both quantitative and qualitative experiments show the effectiveness of our approach compared to other state-of-the-art methods.

<p align="center">
  <img src="assets/tape_teaser.png" width="100%" alt="Overview of the proposed approach">
</p>

Overview of the proposed approach. *Left* given a video, we identify the cleanest frames with CLIP. First, we measure the similarity between the frames and textual prompts that describe different artifacts. Then, we employ Otsu's method to define a threshold for classifying the frames based on their similarity scores, resulting in a set of clean frames. *Right* given a window of $T$ degraded input frames, we select the most similar $D$ clean frames based on the CLIP image features and employ them as references. The proposed Swin-UNet then restores the input frames while effectively leveraging the references.

## Citation

```bibtex
@misc{agnolucci2023referencebased,
    title={Reference-based Restoration of Digitized Analog Videotapes},
    author={Lorenzo Agnolucci and Leonardo Galteri and Marco Bertini and Alberto Del Bimbo},
    year={2023},
    eprint={2310.14926},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## TO-DO:
- [ ] Pre-trained model
- [ ] Testing code
- [ ] Training code
- [ ] Synthetic dataset


## Authors

* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)
* [**Leonardo Galteri**](https://scholar.google.com/citations?user=_n2R2bUAAAAJ&hl=en)
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.com/citations?user=bf2ZrFcAAAAJ&hl=en)

## Acknowledgements

This work was partially supported by the European Commission under European Horizon 2020 Programme, grant number 101004545 - ReInHerit.

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.
