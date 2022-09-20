# Code for the NeuRIPS 2022 paper _A Conditional Randomization Test for Sparse Logistic Regression in High-Dimension_ 


- For more details of our method, see https://arxiv.org/abs/2205.14613 .

- The experiments and methods are developed with Python 3.8, and we recommend
  to use this version of python inside your virtual environment. First install
  the required packages with
  
```
# R and glmnet is needed for rpy2 and running some of the crt-variants
conda install -c conda-forge r-glmnet 
conda install R pip 

pip install -r requirements.txt
```

## Reproduce the experiments

- For Figure 1 and 2 with the qqplot of the decorrelated test statistics and
  D0-CRT test statistics, run
  
```
python exp_qqplot.py
```
  
- For the experiments comparing various methods with varying simulation
  parameters in Figure 3, run
  
```
python exp_simu_varying_parameters.py
```

## Citation

If you use this code or our method in your project, please use the following
citation

```
Nguyen, B. T., Thirion, B., & Arlot, S. (2022). A Conditional Randomization
Test for Sparse Logistic Regression in High-Dimension. NeuRIPS 2022. arXiv preprint arXiv:2205.14613.
```
