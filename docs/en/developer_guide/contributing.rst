Contributing to siirl-agentic
=============================

   **Who this is for:** Anyone who wants to contribute code, documentation, or bug reports to siirl-agentic.

   **What you will get:** Guidelines for development setup, code style, testing, and the pull request process.

Quick Start
-----------

.. code:: bash

   # 1. Clone and setup
   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic

   # 2. Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # 3. Install in development mode
   pip install -e ".[dev]"

   # 4. Install pre-commit hooks
   pre-commit install

Code Style
----------

We enforce consistent code style via pre-commit hooks:

============= =============== ================================
Tool          Purpose         Config
============= =============== ================================
**Black**     Code formatting ``line-length = 140``
**isort**     Import sorting  ``profile = "black"``
**Ruff**      Linting         See ``pyproject.toml``
**mypy**      Type checking   ``python_version = "3.10"``
**codespell** Spelling        Configured in ``pyproject.toml``
============= =============== ================================

All configurations are in ``pyproject.toml``. Pre-commit runs automatically on ``git commit``.

To run manually:

.. code:: bash

   pre-commit run --all-files

Running Tests
-------------

.. code:: bash

   # All tests
   pytest

   # Unit tests only
   pytest tests/unit/

   # Integration tests
   pytest tests/integration/

   # With coverage report
   pytest --cov=siirl --cov-report=html

   # GPU-dependent tests
   pytest -m gpu

   # Specific file
   pytest tests/unit/test_specific.py -v

Test Markers
~~~~~~~~~~~~

============================ =========================
Marker                       Description
============================ =========================
``@pytest.mark.unit``        Fast, isolated unit tests
``@pytest.mark.integration`` Multi-component tests
``@pytest.mark.slow``        Long-running tests
``@pytest.mark.gpu``         Requires GPU hardware
============================ =========================

Pull Request Process
--------------------

1. **Fork** the repository and create a branch from ``main``

2. **Write tests** for your changes

3. **Update documentation** if adding new features

4. **Ensure all checks pass:**

   .. code:: bash

      pre-commit run --all-files
      pytest
      mypy siirl

5. **Write a clear commit message:**

   ::

      Add async multi-turn agent support (#123)

      - Implement async agent executor
      - Add test cases for multi-turn scenarios
      - Update documentation

6. **Submit PR** with a detailed description

What to Contribute
------------------

Bug Reports
~~~~~~~~~~~

- Use GitHub Issues with a minimal reproduction example
- Include: Python version, PyTorch version, CUDA version, Ray version
- Attach relevant log snippets

New Features
~~~~~~~~~~~~

- **New AgentFlow:** See :doc:`Adding New Executor or Flow <adding_new_executor_or_flow>`
- **New Environment:** Subclass ``BasEnvironment`` in ``siirl/environment/``
- **New Reward Function:** Create in ``siirl/utils/reward_score/`` or inject via config
- **Algorithm improvements:** Extend ``siirl/algorithm/``

Documentation
~~~~~~~~~~~~~

- Fix typos, improve examples, add missing content

- Build docs locally to verify:

  .. code:: bash

     cd docs && bash build.sh en

Code Review Checklist
---------------------

Reviewers will check:

- ☐ Code follows project style (Black, isort, Ruff)
- ☐ Tests cover new functionality
- ☐ No performance regressions
- ☐ Documentation updated
- ☐ Type hints added for public APIs
- ☐ Docstrings follow Google style

Project Structure
-----------------

For a detailed walkthrough of the codebase, see:

- :doc:`Code Structure <code_structure>` — Developer-oriented codebase guide
- :doc:`Module Map <../reference/module_map>` — Directory reference

License
-------

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Questions?
----------

- Open an issue on GitHub
- Check existing documentation at `docs/ <../index.rst>`__
