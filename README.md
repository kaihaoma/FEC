# FEC


# Build From Source

1. Install pytorch with CUDA 11.3 . See [pytorch documentation](https://pytorch.org/get-started/locally/)
   ```
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   export CUB_DIR=/usr/local/cuda-11.3/include/cub
   export CUDA_BIN_PATH=/usr/local/cuda-11.3/
   export CUDACXX=/usr/local/cuda-11.3/bin/nvcc
   ```

2. Install Requirements
   ```
   pip install -r requirements.txt
   ```
3. Then, install FBGEMM_GPU from source
   ```
   cd third_party/fbgemm/fbgemm_gpu/
   conda install scikit-build jinja2 ninja cmake hypothesis
   python setup.py install
   ```

4. Next, install FEC (FBGEMM_GPU included) from source (included in third_party folder of torchrec)
   ```
   python setup.py install develop
   ```

5. Run FEC
   ```
   cd example
   python setup.py install develop
   ```
   ## Arguments
   * Systems 
     * `-s <Training Systems>` DLRM training systems, one of {FEC, TorchRec, Parallax, FLECHE}
   * model 
     * `-s <Training Models>` Training models, one of {WDL, DFM, DLRM, DCN}
   * batch size
     * `-b <Training Batch Size>` Training batch size, default setting: 8192
   * datasets
     * `-b <Training datasets>` Training datasets, one of {criteo, avazu, criteo-tb}
   * number of gpu workers
     * `-w <number of gpu workers>` the number of training gpu workers, default setting: 8
   * number of gpu workers
     * `-w <number of gpu workers>` the number of training gpu workers, default setting: 8
   * (Used in FEC only) number of hot entries 
     * `-h <number of hot entries>` the number of hot entries use in FEC, -1 to use the predicted optimal number. default setting: -1
