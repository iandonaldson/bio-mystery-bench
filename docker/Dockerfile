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
    r-base=4.3 \
    bioconductor-deseq2 \
    bioconductor-edger \
    bioconductor-limma \
    r-ggplot2 \
    r-dplyr \
    && micromamba clean -afy

# Python single-cell + proteomics extras
RUN pip install --no-cache-dir \
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

RUN mkdir -p /workspace/data /workspace/scratch /workspace/output
WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
