# Ficaria

Ficaria is a Python package providing custom, **scikit-learn–compatible transformers**
for **data imputation** and **feature selection**.

The transformers are designed to integrate seamlessly with
`scikit-learn` pipelines, making them easy to use in real-world
machine learning workflows and simple to extend for custom needs.

---

## Setup

Ficaria can be installed from PyPI:

```bash
pip install ficaria
```

The package automatically installs all required dependencies, including
NumPy, Pandas, SciPy, and scikit-learn.

---

## Usage

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

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

## Authors

* Aleksandra Kwiatkowska
* Małgorzata Mokwa
* Bogumiła Okrojek

```
