<p align="center">
  <img src="img/logo.png" alt="Project Logo" width="200">
</p>


<h2 align="center"><em>Fuzzy Imputation and Critical Attribute Reduction for Intelligent Analysis</em></h2>

</br>

<p align="center">
  <img src="https://img.shields.io/badge/pypi-v1.0.0-blue?style=flat-square" alt="PyPI">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="license">
  <img src="https://img.shields.io/badge/coverage-96%25-green?style=flat-square" alt="coverage">
</p>


## 🔷 About The Package

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


## ⚙️ Prerequisites

![python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)

The *ficaria* package depends on the following Python libraries:

- **NumPy**
- **Pandas**
- **SciPy**
- **scikit-learn**
- **kneed**


## 🛠 Setup

Ficaria can be installed from PyPI:

```bash
pip install ficaria
```

All dependencies are automatically installed when installing the package via `pip`.


## 🚀 Usage

Ficaria provides scikit-learn–compatible transformers for data imputation and feature selection.
All transformers implement the standard fit / transform interface, so they can be used
directly in pipelines alongside scalers, estimators, and cross-validation tools.

#### Example 1 — Feature Selection with `FuzzyGranularitySelector`

```python
from ficaria import FuzzyGranularitySelector

selector = FuzzyGranularitySelector(n_features=5, eps=0.3)
selector.fit(X_train, y_train)
X_reduced = selector.transform(X_test)
```

#### Example 2 — Data Imputation with `FCMKIterativeImputer`

```python
from ficaria import FCMKIterativeImputer

pipeline.fit(X_train, y_train)
X_transformed = pipeline.transform(X_test)
```

#### Example 3 — Combining Transformers in a Pipeline

Since all transformers implement fit and transform, they can be combined:

```python
from sklearn.pipeline import Pipeline
from ficaria import FuzzyGranularitySelector, FCMKIterativeImputer

pipeline = Pipeline([
    ("imputer", FCMKIterativeImputer()),
    ("selector", FuzzyGranularitySelector(n_features=5, eps=0.3)),
])

pipeline.fit(X_train, y_train)
X_final = pipeline.transform(X_test)
```


## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.


## 👥 Authors

**Aleksandra Kwiatkowska**
Github: <a href="https://github.com/kwiatkowskaa">@kwiatkowskaa</a>

**Małgorzata Mokwa**
Github: <a href="https://github.com/malgosiam2">@malgosiam2</a>

**Bogumiła Okrojek**
Github: <a href="https://github.com/szostkawron">@szostkawron</a>

