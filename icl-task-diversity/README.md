# How does the pretraining distribution shape in-context learning?
This repository is based on the implementation of the paper 
```
Raventós, A., Paul, M., Chen, F., & Ganguli, S. (2023). Pretraining task diversity and the emergence of non-bayesian in-context learning for regression. Advances in neural information processing systems, 36, 14228-14246.
```
and was extended for our purposes.

To run the code, first install the dependencies:

```sh
pip install -r requirements.txt
```

To run an experiment, use the following command:
```sh
python run.py --config-name=icl/configs/generalization_student.yaml
```

The configuration files are located in the `icl/configs` folder. The config files to reproduce the linear regression experiments are:
- `generalization_student.yaml`: Generalization experiments for linear regression with student prior.
- `generalization_gen.yaml`: Generalization experiments for linear regression with generalized normal prior.
- `reweighting_student.yaml`: Reweighting experiments for linear regression with student prior.
- `reweighting_gen.yaml`: Reweighting experiments for linear regression with generalized normal prior.
- `variance.yaml`: Reweighting experiments for linear regression with generalized normal prior and variance analysis. 

The config files to reproduce the OU experiments are:
- `ou_student.yaml`: OU experiments for student prior.
- `ou_gen.yaml`: OU experiments for generalized normal prior.

The config file to reproduce the Volterra experiments is: `volterra.yaml`.


