# siirl-agentic 文档系统方案

## 一、技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 主题 | `sphinx-book-theme` | slime 使用，现代简洁，支持暗色模式，社区活跃 |
| 双语方案 | 分离目录 (en/zh) | slime 模式，维护清晰，可独立构建 |
| 文件格式 | `.rst` 主，`.md` 辅 | 符合要求，兼容 myst-parser |
| 托管 | ReadTheDocs | 自动构建，版本管理，自定义域名 |

---

## 二、目录结构

```
docs/
├── conf.py                    # Sphinx 配置（统一）
├── requirements.txt           # 文档依赖
├── build.sh                   # 单语言构建脚本
├── build_all.sh               # 双语言构建+根索引
├── Makefile                   # 标准 Sphinx Makefile
├── _static/                   # 静态资源（共享）
│   ├── css/
│   │   └── custom.css         # 自定义样式
│   ├── js/
│   │   └── lang-toggle.js     # 语言切换脚本
│   └── image/
│       ├── logo.png           # 项目 Logo
│       └── logo.ico           # Favicon
├── en/                        # 英文文档
│   ├── index.rst              # 英文首页
│   ├── get_started/           # 入门指南
│   ├── programming_guide/     # 编程指南
│   ├── algorithms/            # 算法说明
│   ├── agentic_features/      # Agentic 特性
│   ├── examples/              # 示例教程
│   ├── api/                   # API 参考
│   ├── developer_guide/       # 开发者指南
│   └── faq/                   # 常见问题
└── zh/                        # 中文文档（镜像结构）
    ├── index.rst
    ├── get_started/
    ├── programming_guide/
    ├── algorithms/
    ├── agentic_features/
    ├── examples/
    ├── api/
    ├── developer_guide/
    └── faq/
```

---

## 三、内容规划（参考 slime/verl/siirl）

### 3.1 Get Started（入门指南）
- `installation.rst` - 安装指南（Docker/pip/源码）
- `quickstart.rst` - 快速上手（5分钟跑通）
- `configuration.rst` - 配置说明

### 3.2 Programming Guide（编程指南）
- `architecture.rst` - 系统架构概览
- `code_structure.rst` - 代码结构说明
- `async_training.rst` - 异步训练原理

### 3.3 Algorithms（算法）
- `ppo.rst` - PPO 实现
- `grpo.rst` - GRPO 实现
- `advantage.rst` - 优势估计
- `kl_penalty.rst` - KL 惩罚

### 3.4 Agentic Features（Agentic 特性）- **核心差异化**
- `multi_turn.rst` - 多轮对话训练
- `tool_env.rst` - 工具环境集成
- `swe_bench.rst` - SWE-Bench 集成
- `sandbox_fusion.rst` - 沙箱融合

### 3.5 Examples（示例）
- `ppo_train.rst` - PPO 训练示例
- `grpo_train.rst` - GRPO 训练示例
- `aio_example.rst` - AIO 示例

### 3.6 API Reference（API 参考）
- 自动生成：autodoc/autosummary

### 3.7 Developer Guide（开发者指南）
- `contributing.rst` - 贡献指南
- `debugging.rst` - 调试技巧

### 3.8 FAQ（常见问题）
- 常见错误与解决方案

---

## 四、自定义前端配置

### 4.1 conf.py 核心配置
```python
# 文件: docs/conf.py
project = "siirl-agentic"
html_theme = "sphinx_book_theme"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

html_theme_options = {
    "repository_url": "https://github.com/YOUR_ORG/siirl-agentic",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "show_navbar_depth": 3,
    "show_toc_level": 2,
    "navigation_with_keys": False,
}

# 语言切换支持
language = os.environ.get("SIIRL_DOC_LANG", "en")
```

### 4.2 自定义 CSS 亮点
- 全宽布局（参考 verl）
- 可调整侧边栏宽度
- 代码块优化样式
- 响应式移动端适配

### 4.3 语言切换按钮
- 复用 slime 的 `lang-toggle.js`
- 顶部导航栏显示 EN/中 切换
- localStorage 记住用户偏好

---

## 五、ReadTheDocs 配置

### 5.1 .readthedocs.yaml
```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - requirements: docs/requirements.txt
```

### 5.2 多语言版本托管方案
- 主项目：英文版 (默认)
- 子项目/翻译项目：中文版
- 或：单仓库双构建（通过环境变量 `SIIRL_DOC_LANG`）

---

## 六、实施步骤

### Phase 1: 基础设施（预计 2h）
1. 创建 `docs/` 目录结构
2. 编写 `conf.py` 配置
3. 创建 `requirements.txt`
4. 设置 `_static/` 静态资源
5. 创建构建脚本 `build.sh`, `build_all.sh`

### Phase 2: 框架搭建（预计 1h）
1. 创建 `en/index.rst` 英文首页
2. 创建 `zh/index.rst` 中文首页
3. 创建各章节目录和占位 `.rst` 文件
4. 测试本地构建

### Phase 3: 内容填充（主要工作）
1. 从现有 README/INSTALL/CONTRIBUTING 迁移内容
2. 编写核心章节内容
3. 添加代码示例和图表

### Phase 4: 部署配置（预计 30min）
1. 创建 `.readthedocs.yaml`
2. 配置 ReadTheDocs 项目
3. 设置自定义域名（如需要）

---

## 七、依赖清单

```
# docs/requirements.txt
sphinx>=7.0
sphinx-book-theme>=1.1.0
myst-parser>=2.0
sphinx-copybutton>=0.5
sphinxcontrib-mermaid>=0.9
sphinx-design>=0.5
```

---

## 八、预期效果

- 现代化 UI，支持暗色/亮色模式切换
- 中英文一键切换，用户偏好记忆
- GitHub 集成（编辑、Issue、源码查看）
- 自动 API 文档生成
- 移动端友好
- 代码块一键复制
