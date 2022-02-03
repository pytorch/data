## Building the Documentation

To build the documentation, you will need [Sphinx](http://www.sphinx-doc.org) and the PyTorch theme.

```bash
cd docs/
pip install -r requirements.txt
```

You can then build the documentation by running `make <format>` from the `docs/` folder. Run `make` to get a list of all
available output formats.

```bash
make html
```
