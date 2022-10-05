# Advanced Deep Learning with TensorFlow 2 and Keras

## Hướng dẫn cài đặt

### Cài đặt Miniconda

Trước tiên hãy cài đặt Ubuntu 20.04 hoặc mới nhất, có thể áp dụng được trên cả Ubuntu môi trường WLS2 trên Window.
Sử dụng Termial thực hiện các câu lệnh sau:
```bash
export SETUP_DIR=/u01/setups;\
[ ! -d $SETUP_DIR ] && { sudo mkdir -p $SETUP_DIR; sudo chown -R $(whoami):$(whoami) /u01;};\
export FILE_NAME=Miniconda3-py39_4.12.0-Linux-x86_64.sh;\
[ ! -f $SETUP_DIR/$FILE_NAME ] &&  { curl https://repo.anaconda.com/miniconda/$FILE_NAME -o $SETUP_DIR/$FILE_NAME; };\
[ -f $SETUP_DIR/$FILE_NAME ] && { chmod +x $SETUP_DIR/$FILE_NAME; ls -lt $SETUP_DIR/$FILE_NAME; }

# install mini-anaconda
export INST_DIR=/u01/envs/miniconda;\
  export SETUP_DIR=/u01/setups;\
  export FILE_NAME=Miniconda3-py39_4.12.0-Linux-x86_64.sh;\
  bash $SETUP_DIR/$FILE_NAME -b -p $INST_DIR
```

Sau khi cài đặt xong, thiết lập môi trường cho miniconda

```bash
# config mini-conda
sudo tee /etc/profile.d/minconda3.sh > /dev/null <<'EOF'
  export MINCONDA3_HOME=/u01/envs/miniconda
  export PATH=$PATH:${MINCONDA3_HOME}/bin
EOF
# bash, zsh, csh ...
conda init bash
```
### Thiết lập môi trường cho Tensorflow 2 and Keras

```bash
# To update latest all packages
conda update conda -y
#
# create python 3.9 for data scientist
conda create --name py39ds python=3.9 -y
#
# To activate this environment, use
conda activate py39ds
pip install -r 
```


## code examples
### Chapter 1 - Introduction
* Multilayer Perceptron - MLP on MNIST
* Convolutional Neural Network - CNN on MNIST
* Recurrent Neural Network - RNN on MNIST

