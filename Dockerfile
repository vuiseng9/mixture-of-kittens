FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG MOK_REF=repro

RUN git clone --branch "${MOK_REF}" --single-branch \
	https://github.com/vuiseng9/mixture-of-kittens /workspace/mixture-of-kittens

WORKDIR /workspace/mixture-of-kittens

RUN make install-for-b200

CMD ["/bin/bash"]

