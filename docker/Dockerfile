FROM mambaorg/micromamba:1.5-jammy
USER root

# System build deps + common bioinformatics prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ca-certificates procps \
    build-essential libz-dev libbz2-dev liblzma-dev \
    libcurl4-openssl-dev libssl-dev libncurses5-dev \
    unzip zip \
    && rm -rf /var/lib/apt/lists/*

# Core bioinformatics stack via conda-forge + bioconda
RUN micromamba install -y -n base \
    -c conda-forge -c bioconda -c defaults \
    python=3.11 \
    samtools=1.19 \
    bcftools=1.19 \
    bedtools=2.31 \
    biopython=1.83 \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    pysam \
    star=2.7.11 \
    bowtie2 \
    hisat2 \
    salmon \
    kallisto \
    minimap2 \
    blast \
    r-base=4.3 \
    bioconductor-deseq2 \
    bioconductor-edger \
    bioconductor-limma \
    r-ggplot2 \
    r-dplyr \
    && micromamba clean -afy

# Python single-cell + proteomics extras
# Use absolute path — micromamba base env is at /opt/conda
RUN /opt/conda/bin/pip install --no-cache-dir \
    scanpy \
    anndata \
    pydeseq2 \
    leidenalg \
    python-igraph \
    pyarrow \
    h5py \
    requests \
    biopython

# Keep micromamba activate working for the agent
SHELL ["/bin/bash", "--login", "-c"]

# Expose conda tools on the default PATH for non-login bash shells (e.g. docker exec)
ENV PATH=/opt/conda/bin:$PATH

RUN mkdir -p /workspace/data /workspace/scratch /workspace/output /workspace/skills
WORKDIR /workspace

# Bio method reference recipes. Source of truth lives at SKILLS/<name>/SKILL.md
# in the repo root — these copies are baked into /workspace/skills/ so the agent
# can `cat` them in-container without needing internet access.
# NOTE: requires `docker build -f docker/Dockerfile .` from the repo root (not
# `docker build docker/`); see scripts/smoke_test_container.py for the canonical
# rebuild command.
COPY SKILLS/deg-functional-enrichment/SKILL.md /workspace/skills/deg-functional-enrichment.md
COPY SKILLS/chipseq-tf-identification/SKILL.md /workspace/skills/chipseq-tf-identification.md
COPY SKILLS/genome-retrieval/SKILL.md /workspace/skills/genome-retrieval.md
COPY SKILLS/ucsc-sequence-fetch/SKILL.md /workspace/skills/ucsc-sequence-fetch.md
COPY SKILLS/blast-search/SKILL.md /workspace/skills/blast-search.md

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
