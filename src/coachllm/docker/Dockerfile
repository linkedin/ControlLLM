FROM container-image-registry.corp.linkedin.com:8083/temp/afatahib/mlperf-pt:ibs
# FROM container-image-registry.corp.linkedin.com:8083/lps-image/linkedin/mldev-images/mldev-torch-image:0.0.157
USER root
RUN source /opt/rh/devtoolset-8/enable
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
# wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-rhel7-12-2-local-12.2.0_535.54.03-1.x86_64.rpm
RUN rpm -i cuda-repo-rhel7-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
# RUN rpm -i cuda-repo-rhel7-12-2-local-12.2.0_535.54.03-1.x86_64.rpm
# RUN yum-config-manager --disable grid
RUN yum clean all && yum -y install nvidia-driver-latest-dkms && yum -y install cuda
USER jobuser
RUN source scl_source enable devtoolset-11 && pip install torch==2.1.2
RUN pip install packaging && pip uninstall -y ninja && pip install ninja && pip install wheel setuptools &&  source scl_source enable devtoolset-11 && pip install --extra-index-url https://pypi.org/simple flash-attn==2.1.0 --no-build-isolation

RUN pip install tensorflow==2.11.0.16
RUN pip install protobuf==4.21.10
# RUN pip install scikit-learn==0.24.2
# RUN pip install transformers==4.11.3
# RUN pip install spacy==3.7.2
RUN pip install joblib==1.1.0
RUN pip install torchsummary==1.5.1
RUN pip install submitit==1.4.2
# RUN pip install timm==0.3.2
# RUN pip install mmcv-full==1.7.1
RUN pip install cython==0.29.24
RUN pip install einops==0.4.1
# RUN pip install albumentations==1.1.0
# RUN pip install cityscapesscripts==2.2.0
RUN pip install imagecorruptions==1.1.2
# RUN pip install mmlvis==10.5.3
# RUN pip install scipy==1.5.4
# RUN pip install sklearn==0.0
RUN pip install asynctest
RUN pip install codecov
RUN pip install interrogate
RUN pip install isort==4.3.21
# Note: used for kwarray.group_items, this may be ported to mmcv in the future.
# RUN pip install kwarray
# RUN pip install ubelt
# RUN pip install xdoctest
# RUN pip install yapf
RUN pip install matplotlib==3.3.4
# RUN pip install mmpycocotools==12.0.3
RUN pip install six==1.15.0
RUN pip install terminaltables
# RUN pip install protobuf==3.16.0

# Note: for LLM
RUN pip install black==23.1.0
RUN pip install datasets==2.14.6
# RUN pip install deepspeed==0.12.2
RUN pip install einops>=0.6.1
RUN pip install evaluate==0.4.0
RUN pip install flake8>=6.0.0
RUN pip install hf-doc-builder>=0.4.0
RUN pip install huggingface_hub==0.19.4
RUN pip install isort>=5.12.0
RUN pip install ninja>=1.11.1
RUN pip install numpy>=1.24.2
RUN pip install packaging>=23.0
RUN pip install parameterized>=0.9.0
RUN pip install peft==0.6.1
RUN pip install protobuf==3.20.1  # Needed to avoid conflicts with `transformers`
RUN pip install pytest==7.4.3
RUN pip install safetensors>=0.3.3
RUN pip install scipy==1.8.0
RUN pip install tensorboard==2.11.2
# RUN pip install torch==2.1.0
RUN pip install transformers==4.37.0
RUN pip install SentencePiece==0.1.98  # Needed for LlamaTokenizer in transformers
RUN pip install accelerate==0.26.1
# RUN pip install accelerate==0.26.1
RUN pip install bitsandbytes==0.41.2.post2
# RUN pip install bitsandbytes==0.42.0
RUN pip install peft==0.7.1  # needed for PE fine tuning
RUN pip install trl==0.7.4
RUN pip install py7zr==0.20.5
RUN pip install fire==0.4.0
RUN pip install torch-tb-profiler==0.4.1
RUN pip install ipywidgets==8.0.4
RUN pip install datasets==2.16.1
RUN pip install jinja2>=3.0.0
RUN pip install tqdm>=4.64.1
RUN pip install pydantic-settings==2.0.3

# important torch and flash-attn
RUN pip install torch==2.1.2
RUN pip install flash-attn==2.4.2
RUN pip install torchvision==0.16.0+cu118 torchaudio==2.1.0
RUN pip install deepspeed==0.13.1
