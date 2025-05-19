# Collaborative Inference With a Secure Ensemble

This repository provides implementations for experimenting with secure ensemble based on multi-key homomorphic encryption. It includes simulation experiments for both logistic regression and CNN models.

## Requirements

* Go 1.22 or higher
* Python 3.8 or higher (for pre-training)

## Installation

1. Clone the repository with submodules:

```bash
git clone --recursive https://github.com/anony-submit/secure-ensemble.git
cd secure-ensemble
```

2. Install and tidy up dependencies:

```bash
# For logistic regression experiments
cd logistic_regression/go_inference
go mod download
go mod tidy

# CNN experiments (repeat for each model)
cd cnn/mnist/go_inference
go mod download
go mod tidy

cd cnn/fmnist/go_inference
go mod download
go mod tidy

cd cnn/cifar10/go_inference
go mod download
go mod tidy

cd cnn/svhn/go_inference
go mod download
go mod tidy

# For shared utility packages
cd ../../../pkg
go mod download
go mod tidy

cd ../snu-mghe
go mod download
go mod tidy
```

## Project Structure

```
.
├── baseline/                # FedAvg and FedProx baseline experiments
│   ├── cnn/                 # CNN baselines using FedGen [3] (submodule)
│   └── logistic/            # Baseline logistic regression models
├── cnn/                     # Secure inference experiments with CNNs
├── logistic_regression/     # Secure inference experiments with logistic regression
├── pkg/                     # Shared Go modules (e.g., activation functions, serialization)
├── snu-mghe/                # Multi-key homomorphic encryption scheme [2] (submodule)
```

### `cnn/`

Secure ensemble inference for CNN models on:

* MNIST
* FMNIST
* CIFAR10
* SVHN

Each dataset directory contains:

* `py_pretrain/`: Model training scripts (Python)
* `go_inference/`: Secure ensemble implementation and experiments (Go)

Each experiment logs latency and accuracy results to `.txt` files.

**To run experiments**, navigate to each experiment directory and execute:

```bash
# CIFAR10
cd cnn/cifar10/go_inference/experiment
go test -v -timeout 0 -run TestCIFAR10Dropout
go test -v -timeout 0 -run TestCIFAR10Ensemble

# FMNIST
cd cnn/fmnist/go_inference/experiment
go test -v -timeout 0 -run TestFMNISTDropout
go test -v -timeout 0 -run TestFMNISTEnsemble

# MNIST
cd cnn/mnist/go_inference/experiment
go test -v -timeout 0 -run TestMNISTDropout
go test -v -timeout 0 -run TestMNISTEnsemble

# SVHN
cd cnn/svhn/go_inference/experiment
go test -v -timeout 0 -run TestSVHNEnsemble
```

### `logistic_regression/`

Implements secure inference for logistic regression models.

* `py_pretrain/`: Model training 
* `go_inference/`: Secure ensemble implementation

Each experiment logs latency and accuracy results to `.txt` files.

**To run experiments**, navigate to the experiment directory and execute:

```bash
cd logistic_regression/go_inference/experiments

# Breast Cancer (WDBC)
go test -timeout 0 -v -run TestInferenceWDBC

# Heart Disease (HD)
go test -timeout 0 -v -run TestInferenceHeartDisease

# PIMA Diabetes (PID)
go test -timeout 0 -v -run TestInferencePima

# Run all tests
go test -timeout 0 -v
```

### `baseline/`

Baseline accuracy comparisons using FedAvg and FedProx. The `cnn/` subdirectory includes implementations sourced from the FedGen submodule, which provides FedAvg and FedProx for CNNs.

FedGen GitHub: [https://github.com/zhuangdizhu/FedGen](https://github.com/zhuangdizhu/FedGen)

### `pkg/`

Shared Go modules used across logistic and CNN experiments:

* `activation/`: Polynomial evaluation for approximated activations 
* `logistic/`: Batched inference for logistic regression
* `serialization/`: Key and ciphertext serialization

### `snu-mghe/`

Submodule for multi-key homomorphic encryption schemes.

GitHub: [https://github.com/SNUCP/snu-mghe](https://github.com/SNUCP/snu-mghe)

## References

[1] H. Chen, W. Dai, M. Kim, and Y. Song, Efficient Multi-Key Homomorphic Encryption with Packed Ciphertexts with Application to Oblivious Neural Network Inference,  
Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 395–412, 2019.

[2] T. Kim, H. Kwak, D. Lee, J. Seo, and Y. Song, Asymptotically Faster Multi-Key Homomorphic Encryption from Homomorphic Gadget Decomposition,  
Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (CCS), pp. 726–740, 2023.

[3] Z. Zhu, J. Hong, and J. Zhou, Data-Free Knowledge Distillation for Heterogeneous Federated Learning,  
Proceedings of the 38th International Conference on Machine Learning (ICML), pp. 12878–12889, 2021.


## License

See the [LICENSE](LICENSE) file for details.
