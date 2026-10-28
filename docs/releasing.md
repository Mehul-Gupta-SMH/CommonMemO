# Releasing to PyPI

What is checked and what is left to the owner ([#12](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/12), [#103](https://github.com/Mehul-Gupta-SMH/CommonMemO/issues/103)).

## Checked (2026-10-28, v0.0.1)

- `commonmemo` is free on PyPI.
- `python -m build` makes an sdist and a wheel; `twine check` passes on both. The wheel holds the 47 modules and
  `opencode_plugin.js`, no tests or scripts.
- A fresh virtualenv on Python 3.11, 3.12 and 3.13 installs the wheel (`pip check` clean), runs the quickstart,
  `memory://`, `commonmemo show / hook / setup / mcp` and the inspector, and importing the library loads no
  third-party module. The whole test suite passes against the installed wheel. Python 3.10 runs in CI.
- Every extra's dependencies exist on PyPI and resolve together.
- The README is rewritten at build time so its images and links point at GitHub (`hatch-fancy-pypi-readme`
  in `pyproject.toml`); the README in the repository is unchanged.

## What only the owner can do

1. **PyPI account** with two-factor authentication.
2. **Register the project without a token** (trusted publishing): on PyPI, *Your projects → Publishing → Add a new
   pending publisher*: project `commonmemo`, owner `Mehul-Gupta-SMH`, repository `CommonMemO`, workflow
   `release.yml`, environment `pypi`.
3. **GitHub environment** `pypi` (*Settings → Environments*); add yourself as a required reviewer if you want to
   approve each publish.
4. **Decide the version.** It is in two places, `pyproject.toml` and `src/commonmemo/__init__.py`; a test keeps
   them equal. `0.0.1` is marked Alpha in the classifiers. The CHANGELOG's "0.0.1 — unreleased" heading becomes the
   release date.
5. **Release:** tag `v0.0.1` and publish a GitHub release for it. `release.yml` builds, checks that the tag names the
   version in `pyproject.toml`, runs `twine check` and uploads.

An upload cannot be undone: a version, once published, cannot be replaced (it can be yanked). To rehearse, point a
second pending publisher at TestPyPI and run the workflow there first.

## Before each release

```
python -m build && twine check dist/*
python -m venv /tmp/fresh && /tmp/fresh/bin/pip install dist/*.whl && /tmp/fresh/bin/python -m pytest -q tests
pytest -q && COMMONMEMO_TEST_INTERFACE=1 pytest -q          # and the Postgres run, see CI
```
