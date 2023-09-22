FROM jupyter/scipy-notebook:python-3.9

USER root

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends openjdk-17-jre && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install requests tabulate future matplotlib
RUN pip uninstall h2o
RUN pip install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o

USER ${NB_UID}