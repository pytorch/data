## Building the Documentation

To build the documentation, you will need [Sphinx](http://www.sphinx-doc.org) and the PyTorch theme.

```bash
cd docs/
pip install -r requirements.txt
```

You can then build the documentation by running `make <format>` from the `docs/` folder. Run `make` to get a list of all
available output formats. Run

```bash
make html
```

to build the documentation. The html files can then be found in `build/html`. To validate the code examples use:

```bash
make doctest
```

Note that currently only code-output-style blocks are tested as many standard reST doctest examples do not work atm. The
results can then be found in `build/html/output.txt`. To also test interactive Python sessions you can temporarily
replace `doctest_test_doctest_blocks` in
[`source/conf.py`](https://github.com/pytorch/data/blob/main/docs/source/conf.py) with a non-empty string.

## Improving the Documentation

Feel free to open an issue or pull request to inform us of any inaccuracy or potential improvement that we can make to
our documentation.
