# Collaborative Inference With a Secure Ensemble on Locally Trained Models

This repository provides implementations for experimenting with secure ensemble based on multi-key homomorphic encryption. It includes simulation experiments for both logistic regression and CNN models.

## Requirements

- Go 1.22 or higher
- Python 3.8 or higher (for pre-training)

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/anony-submit/secure-ensemble.git
cd secure-ensemble
```

2. Install and tidy up dependencies:
```bash
go mod download  # Download dependencies
go mod tidy     # Ensure dependencies are properly aligned
```

## Project Structure

The project consists of two main components:
- CNN implementations (MNIST, FMNIST, CIFAR10, SVHN)
- Logistic Regression implementation

Each model implementation contains:
- `py_pretrain/`: Code for training the models
- `go_inference/`: Code for secure ensemble experiments, measuring latency and accuracy

## Usage

To run secure ensemble experiments, modify the configuration in the experiment directory and run the test. For example:

```bash
cd cnn/svhn/go_inference/experiment
go test -v -timeout 0 -run TestSVHNEnsemble
```

## References

This project uses the multi-key homomorphic encryption scheme introduced in [1, 2], with the implementation from [2].

[1] H. Chen, W. Dai, M. Kim, and Y. Song, "Efficient Multi-Key Homomorphic Encryption with Packed Ciphertexts with Application to Oblivious Neural Network Inference," in Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security (CCS '19), 2019, pp. 395-412.

[2] T. Kim, H. Kwak, D. Lee, J. Seo, and Y. Song, "Asymptotically Faster Multi-Key Homomorphic Encryption from Homomorphic Gadget Decomposition," in Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security (CCS '23), 2023, pp. 726-740.

## License

See the [LICENSE](LICENSE) file for details.
