# Data sources

## DSRL

The 38-task catalog is distributed by [DSRL](https://github.com/liuzuxin/DSRL), with the files used here hosted in the [DSRL dataset repository](https://huggingface.co/datasets/YYY-45/DSRL). `manifests/datasets.json` records the task names, original URLs, exact file sizes, and SHA256 values audited for these experiments. The downloader checks the expected size and hash before accepting a file.

The DSRL software repository uses Apache-2.0. Dataset access and reuse follow the publisher's dataset terms; the OTTP software license does not assign a new license to upstream data.

## FISOR Point Robot

The [FISOR project](https://zhengyinan-air.github.io/FISOR/) links its Point Robot dataset through an [author-hosted share](https://cloud.tsinghua.edu.cn/d/162d6fe92bde43e28676/). The recorded file is `point_robot-expert-random-100k.hdf5`, 10,904,096 bytes, SHA256 `20773a5530cbda57644adac7226812cf492593a617735b4685a969131d37c470`.

The source loader at commit `2097e8bbc96ef49183f22cb2a1b05948b5859ec0` maps `h` to the training cost and reads `cost` as the binary violation indicator. The adaptation uses `done` as the episode boundary and drops 22 transitions in the incomplete final episode. The source and field audits are in `results/external/`.

The publisher provides research-download instructions. A separate data redistribution license was not identified; this archive supplies links, hashes, adaptation code, and results.

## Local files

Place downloads and generated precursor files in a working directory. Archived split indices refer to the original row order, so dataset hashes must match before using those indices. Trained weights and generated precursor tensors are created by the reproduction commands.
