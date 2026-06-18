# Building and Releasing a New Version of `splinecloud-scipy`


## 1. Check Virtual Environment Setup

Maintain two separate virtual environments:

* **Work environment** (e.g. `.workvenv`) for package development.
* **Test environment** (e.g. `.testenv`) for validating package installation from TestPyPI and PyPI.

### Work Environment

The work environment should contain the local project installed in editable mode:

```bash
source .workvenv/bin/activate
pip install -e .
```

Verify that Python imports the package from the local source tree:

```bash
python -c "import splinecloud_scipy; print(splinecloud_scipy.__file__)"
```

The reported path should point to the repository directory.

### Test Environment

The test environment should be kept clean and should **not** contain an editable installation of the package.

Create it if necessary:

```bash
python -m venv .testenv
source .testenv/bin/activate
pip install --upgrade pip
```

Use this environment exclusively to verify installation of released package versions from TestPyPI and PyPI.


## 2. Activate the work environment

```bash
source .workvenv/bin/activate
```

## 3. Update the package version

Update the version in `pyproject.toml`.

Example:

```TOML
version = "1.x.y"
```

## 4. Commit changes

```bash
git add pyproject.toml
git commit -m "Elevate version to v1.x.y"
git push origin main
```

## 5. Create and push a Git tag

```bash
git tag v1.x.y
git push origin v1.x.y
```

Verify:

```bash
git ls-remote --tags origin
```

## 6. Build the package

Ensure working directory is clean before building:

```bash
rm -rf build dist *.egg-info
```

Build package from the work environment:

```bash
python -m build
```

Expected output:

```text
dist/
├── splinecloud_scipy-1.x.y.tar.gz
└── splinecloud_scipy-1.x.y-py3-none-any.whl
```

## 7. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

## 8. Test installation from TestPyPI (from test environment)

Create a clean virtual environment:

```bash
deactivate
source .testenv/bin/activate
```

Install the package from TestPyPI:

```bash
pip install \
  -i https://test.pypi.org/simple \
  --extra-index-url https://pypi.org/simple \
  splinecloud-scipy==1.x.y
```

Verify:

```bash
python -c "import splinecloud_scipy; print(splinecloud_scipy.__version__)"
```

## 9. Upload to PyPI (from work environment)

After successful testing:

```bash
deactivate
source .workvenv/bin/activate
twine upload dist/*
```

## 10. Verify installation from PyPI (from test environment)

```bash
deactivate
source .testenv/bin/activate
pip install --upgrade splinecloud-scipy
python -c "import splinecloud_scipy; print(splinecloud_scipy.__version__)"
```

## Notes

* Always increase the version number before building.
* A version already uploaded to PyPI/TestPyPI cannot be uploaded again.
* If a release contains errors, create a new version (e.g. `1.X.Z`) instead of reusing an existing version number.
* Keep virtual environments (`.workvenv`, `.testvenv`) outside package discovery or exclude them from packaging.
