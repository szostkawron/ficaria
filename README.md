<p>
  <img src="https://img.shields.io/badge/pypi-v1.0.0-blue?style=flat-square" alt="PyPI">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="license">
  <img src="https://img.shields.io/badge/coverage-96%25-green?style=flat-square" alt="coverage">
</p>


<p align="center">
  <img src="img/logo.png" alt="Project Logo" width="250">
</p>


<h2 align="center"><em>Fuzzy Imputation and Critical Attribute Reduction for Intelligent Analysis</em></h2>

</br>


## 📝 About The Package

The *ficaria* package is a Python package providing custom, **scikit-learn–compatible transformers**
for **data imputation** and **feature selection**. The transformers are designed to integrate seamlessly with
`scikit-learn` pipelines, making them easy to use in real-world
machine learning workflows and straightforward to extend for
custom or research-oriented use cases.

The package was developed as part of a **Bachelor’s degree thesis**
at the **Warsaw University of Technology**, Faculty of
**Mathematics and Information Science**. All implemented methods are **fuzzy-based**, leveraging concepts
from **fuzzy set theory** to handle uncertainty, vagueness, and
incomplete data in a principled and interpretable manner.
This makes *ficaria* particularly suitable for datasets where
classical crisp methods may be insufficient or overly restrictive.


## Prerequisites

![python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)

The *ficaria* package depends on the following Python libraries:

- **NumPy** ≥ 1.26.4
- **Pandas** ≥ 2.1.4
- **SciPy** ≥ 1.11.4
- **scikit-learn** ≥ 1.4.0
- **kneed** ≥ 0.8.5

All dependencies are automatically installed when installing the package via `pip`.


## Setup

Ficaria can be installed from PyPI:

```bash
pip install ficaria
```


## Usage

```python
from ficaria import FuzzyGranularitySelector

selector = FuzzyGranularitySelector(n_features=5, eps=0.3)
selector.fit(X, y)
X_reduced = selector.transform(X)
```

Ficaria transformers follow the scikit-learn API and can be used directly
in pipelines.

Example:

```python
from sklearn.pipeline import Pipeline
from ficaria import FCMKIterativeImputer

pipeline = Pipeline(
    steps=[
        ("transformer", FCMKIterativeImputer()),
    ]
)

pipeline.fit(X_train, y_train)
X_transformed = pipeline.transform(X_test)
```

Because all transformers implement `fit` and `transform`, they can be
combined with other scikit-learn components such as scalers, estimators,
and cross-validation tools.

Refer to the package documentation and docstrings for detailed usage
examples of individual transformers.


## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.


## Authors

* Aleksandra Kwiatkowska
* Małgorzata Mokwa
* Bogumiła Okrojek

