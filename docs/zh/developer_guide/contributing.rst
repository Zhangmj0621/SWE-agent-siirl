贡献指南
========

   **适合谁：** 想要为 siirl-agentic 贡献代码、文档或 bug 报告的任何人。

   **你将获得：** 开发环境搭建、代码风格、测试和 PR 流程的完整指南。

快速开始
--------

.. code:: bash

   # 1. 克隆并初始化
   git clone https://github.com/sii-research/siirl-agentic.git
   cd siirl-agentic

   # 2. 创建虚拟环境
   python -m venv venv
   source venv/bin/activate

   # 3. 开发模式安装
   pip install -e ".[dev]"

   # 4. 安装 pre-commit hooks
   pre-commit install

代码风格
--------

我们通过 pre-commit hooks 强制执行统一的代码风格：

============= =============== ================================
工具          用途            配置
============= =============== ================================
**Black**     代码格式化      ``line-length = 140``
**isort**     import 排序     ``profile = "black"``
**Ruff**      代码检查        见 ``pyproject.toml``
**mypy**      类型检查        ``python_version = "3.10"``
**codespell** 拼写检查        在 ``pyproject.toml`` 中配置
============= =============== ================================

所有配置均在 ``pyproject.toml`` 中。Pre-commit 在 ``git commit`` 时自动运行。

手动运行：

.. code:: bash

   pre-commit run --all-files

运行测试
--------

.. code:: bash

   # 所有测试
   pytest

   # 仅单元测试
   pytest tests/unit/

   # 集成测试
   pytest tests/integration/

   # 生成覆盖率报告
   pytest --cov=siirl --cov-report=html

   # 依赖 GPU 的测试
   pytest -m gpu

   # 指定文件
   pytest tests/unit/test_specific.py -v

测试标记
~~~~~~~~

============================ =========================
标记                         说明
============================ =========================
``@pytest.mark.unit``        快速单元测试
``@pytest.mark.integration`` 多组件测试
``@pytest.mark.slow``        长耗时测试
``@pytest.mark.gpu``         需要 GPU 硬件
============================ =========================

PR 流程
-------

1. **Fork** 仓库，从 ``main`` 创建分支

2. 为你的修改**编写测试**

3. 如果添加了新功能，**更新文档**

4. **确保所有检查通过：**

   .. code:: bash

      pre-commit run --all-files
      pytest
      mypy siirl

5. **编写清晰的 commit 信息：**

   ::

      Add async multi-turn agent support (#123)

      - Implement async agent executor
      - Add test cases for multi-turn scenarios
      - Update documentation

6. **提交 PR**，附上详细说明

可以贡献什么
------------

Bug 报告
~~~~~~~~

- 在 GitHub Issues 中提交，附上最小可复现示例
- 包含：Python 版本、PyTorch 版本、CUDA 版本、Ray 版本
- 附上相关日志片段

新功能
~~~~~~

- **新 AgentFlow：** 参见 :doc:`新增 Executor 或 Flow <adding_new_executor_or_flow>`
- **新环境：** 在 ``siirl/environment/`` 中继承 ``BasEnvironment``
- **新奖励函数：** 在 ``siirl/utils/reward_score/`` 中创建或通过配置注入
- **算法改进：** 扩展 ``siirl/algorithm/``

文档
~~~~

- 修复错别字、改进示例、补充缺失内容

- 本地构建验证：

  .. code:: bash

     cd docs && bash build.sh en

代码审查 Checklist
------------------

审查者将检查：

- ☐ 代码遵循项目风格（Black、isort、Ruff）
- ☐ 测试覆盖了新功能
- ☐ 没有性能回退
- ☐ 文档已更新
- ☐ 公共 API 添加了类型注解
- ☐ Docstring 遵循 Google 风格

License
-------

提交贡献即表示你同意你的贡献将以 Apache License 2.0 授权。
