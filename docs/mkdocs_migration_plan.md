# siirl-agentic 文档美化方案：迁移到 MkDocs Material

> 目标：将文档视觉质量从当前 Sphinx 水平提升到 ROLL (https://alibaba.github.io/ROLL/) 同等级别。

---

## 一、当前问题诊断

### 1.1 与 ROLL 的视觉对比

| 维度 | ROLL (Docusaurus) | siirl-agentic (Sphinx) | 差距 |
|------|-------------------|----------------------|------|
| **落地页** | Hero 区 + 特性卡片 + CTA 按钮 + 统计数据 | 纯 RST 文本 + 内联 HTML 按钮 | 巨大 |
| **侧边栏** | 分组折叠，2-3 层级深度 | 30+ 项平铺列表，无层级 | 巨大 |
| **暗色模式** | 默认暗色，精心设计的配色 | 基础暗色，配色不统一 | 大 |
| **导航** | 面包屑 + 上/下篇按钮 + 顶部 Tab | 仅侧边栏 | 大 |
| **搜索** | 即时搜索 + 高亮 | Sphinx 基础搜索 | 中 |
| **代码块** | 复制按钮 + 行号 + 高亮行 | 复制按钮（已有） | 小 |
| **响应式** | 完善的移动端适配 | 基础适配 | 中 |

### 1.2 根本原因

不是内容问题（已有 63 个 RST 文件，含 25 个 Mermaid 图），而是**框架代差**：
- `sphinx_book_theme` 是学术风格主题，视觉上限有限
- Sphinx 的 toctree 机制导致侧边栏天然扁平
- 没有现代落地页能力

---

## 二、框架选型

### 2.1 候选方案对比

| 维度 | Docusaurus | MkDocs Material | VitePress | Sphinx (现状) |
|------|-----------|----------------|-----------|--------------|
| 语言生态 | Node.js/React | **Python/pip** | Node.js/Vue | Python |
| 视觉质量 | 9/10 | **8.5/10** | 9/10 | 5/10 |
| 迁移成本 | 高 | **中** | 高 | 无 |
| 暗色模式 | 内置 | **内置** | 内置 | 基础 |
| 折叠侧边栏 | 内置 | **内置** | 内置 | 差 |
| i18n | 需 JSON 翻译 | **目录映射** | 目录映射 | 环境变量 |
| Mermaid | 插件 | **原生** | 插件 | 插件 |
| 团队熟悉度 | 低 | **高** | 低 | 高 |

### 2.2 推荐：MkDocs Material

**理由：**
1. **Python 生态**：`pip install` 即可，不引入 Node.js 依赖
2. **视觉质量足够**：Material Design 3，内置暗色模式、折叠导航、搜索、代码高亮
3. **现有目录结构直接映射**：`en/` + `zh/` 目录 → i18n 插件原生支持
4. **Mermaid 零改动**：25 个图只需改 `.. mermaid::` 为 ` ```mermaid ` 围栏
5. **Markdown 更通用**：RST → MD 一次转换，降低长期维护门槛

---

## 三、迁移后目录结构

```
siirl-agentic/docs/
├── mkdocs.yml                        # 主配置文件
├── requirements-docs.txt             # 文档依赖
├── overrides/                        # 主题覆盖
│   ├── home.html                     # 自定义落地页模板
│   └── stylesheets/
│       └── extra.css                 # 自定义样式
├── docs/                             # 英文内容（默认语言）
│   ├── index.md                      # 落地页（使用 home.html）
│   ├── highlights/
│   │   ├── index.md                  # Section 首页
│   │   ├── why-agentic-rl.md
│   │   ├── native-agentic-trajectory-training.md
│   │   ├── pluggable-agentflow-protocol.md
│   │   ├── mpmd-async-execution-engine.md
│   │   └── aio-elastic-tool-infrastructure.md
│   ├── concepts/
│   │   ├── design-philosophy.md
│   │   ├── architecture-overview.md
│   │   └── async-training-lifecycle.md
│   ├── get-started/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── first-agentic-training-job.md
│   ├── user-guide/
│   │   ├── configuration-system.md
│   │   ├── ppo-training.md
│   │   ├── grpo-training.md
│   │   ├── agentic-multiturn.md
│   │   ├── tool-env-and-swe.md
│   │   ├── aio-tool-infrastructure.md
│   │   ├── deployment-modes.md
│   │   ├── validate-reuse-and-eval-scaling.md
│   │   ├── checkpoint-resume.md
│   │   └── metrics-and-evaluation.md
│   ├── advanced/
│   │   ├── performance-tuning.md
│   │   └── failure-propagation.md
│   ├── reference/
│   │   ├── config-reference.md
│   │   ├── module-map.md
│   │   └── compatibility-matrix.md
│   ├── developer-guide/
│   │   ├── code-structure.md
│   │   ├── adding-new-executor-or-flow.md
│   │   └── contributing.md
│   └── faq/
│       └── troubleshooting.md
├── docs_zh/                          # 中文内容（镜像 docs/ 结构）
│   ├── index.md
│   └── ...（同上结构）
├── scripts/
│   └── convert-rst-to-md.sh          # RST → MD 批量转换脚本
└── _legacy_sphinx/                   # 旧 Sphinx 文件备份（迁移验证后删除）
    ├── conf.py
    ├── en/
    ├── zh/
    └── ...
```

---

## 四、核心配置模板

### 4.1 mkdocs.yml

```yaml
site_name: siirl-agentic
site_url: https://sii-research.github.io/siirl-agentic
site_description: "Asynchronous Multi-Turn Agentic RL Training Framework"
repo_url: https://github.com/sii-research/siirl-agentic
repo_name: sii-research/siirl-agentic
edit_uri: edit/main/docs/docs/

theme:
  name: material
  custom_dir: overrides
  language: en
  palette:
    # 暗色模式（默认，对标 ROLL）
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to light mode
    # 亮色模式
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to dark mode
  font:
    text: Inter
    code: JetBrains Mono
  features:
    - navigation.instant          # SPA 式页面切换
    - navigation.tracking         # URL 跟踪锚点
    - navigation.sections         # 侧边栏分组折叠
    - navigation.expand           # 自动展开当前 section
    - navigation.indexes          # Section 索引页
    - navigation.top              # 返回顶部按钮
    - navigation.tabs             # 顶部导航 Tab
    - navigation.tabs.sticky      # Tab 固定
    - content.code.copy           # 代码复制按钮
    - content.code.annotate       # 代码注解
    - content.tabs.link           # 关联 Tab 切换
    - search.suggest              # 搜索建议
    - search.highlight            # 搜索高亮
    - toc.follow                  # TOC 跟随滚动
  icon:
    repo: fontawesome/brands/github

# 侧边栏导航结构（分组折叠，对标 ROLL）
nav:
  - Home: index.md
  - Highlights:
    - highlights/index.md
    - Why Agentic RL: highlights/why-agentic-rl.md
    - Native Trajectory Training: highlights/native-agentic-trajectory-training.md
    - AgentFlow Protocol: highlights/pluggable-agentflow-protocol.md
    - MPMD Async Engine: highlights/mpmd-async-execution-engine.md
    - AIO Tool Infrastructure: highlights/aio-elastic-tool-infrastructure.md
  - Concepts:
    - Design Philosophy: concepts/design-philosophy.md
    - Architecture Overview: concepts/architecture-overview.md
    - Async Training Lifecycle: concepts/async-training-lifecycle.md
  - Get Started:
    - Installation: get-started/installation.md
    - Quickstart: get-started/quickstart.md
    - First Agentic Job: get-started/first-agentic-training-job.md
  - User Guide:
    - Configuration: user-guide/configuration-system.md
    - PPO Training: user-guide/ppo-training.md
    - GRPO Training: user-guide/grpo-training.md
    - Agentic Multi-turn: user-guide/agentic-multiturn.md
    - Tool Env & SWE: user-guide/tool-env-and-swe.md
    - AIO Infrastructure: user-guide/aio-tool-infrastructure.md
    - Deployment Modes: user-guide/deployment-modes.md
    - Validate & Eval: user-guide/validate-reuse-and-eval-scaling.md
    - Checkpoint & Resume: user-guide/checkpoint-resume.md
    - Metrics: user-guide/metrics-and-evaluation.md
  - Advanced:
    - Performance Tuning: advanced/performance-tuning.md
    - Failure Propagation: advanced/failure-propagation.md
  - Reference:
    - Config Reference: reference/config-reference.md
    - Module Map: reference/module-map.md
    - Compatibility: reference/compatibility-matrix.md
  - Developer Guide:
    - Code Structure: developer-guide/code-structure.md
    - Custom Flows: developer-guide/adding-new-executor-or-flow.md
    - Contributing: developer-guide/contributing.md
  - FAQ: faq/troubleshooting.md

# Markdown 扩展
markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.critic
  - pymdownx.caret
  - pymdownx.keys
  - pymdownx.mark
  - pymdownx.tilde
  - tables
  - attr_list
  - md_in_html
  - def_list
  - footnotes
  - toc:
      permalink: true

# 插件
plugins:
  - search:
      lang:
        - en
        - zh
  - i18n:
      default_language: en
      languages:
        - locale: en
          name: English
          default: true
          build: true
        - locale: zh
          name: 中文
          build: true
          nav_translations:
            Home: 首页
            Highlights: 亮点
            Concepts: 核心概念
            Get Started: 快速开始
            User Guide: 用户指南
            Advanced: 进阶
            Reference: 参考
            Developer Guide: 开发者指南
            FAQ: 常见问题

# 语言切换（替代现有 lang-toggle.js）
extra:
  alternate:
    - name: English
      link: /en/
      lang: en
    - name: 中文
      link: /zh/
      lang: zh
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/sii-research/siirl-agentic

extra_css:
  - overrides/stylesheets/extra.css
```

### 4.2 requirements-docs.txt

```
mkdocs>=1.6
mkdocs-material>=9.5
mkdocs-static-i18n>=1.2
pymdown-extensions>=10.0
```

---

## 五、落地页设计

### 5.1 HTML 模板（overrides/home.html）

```html
{% extends "main.html" %}

{% block tabs %}
  {{ super() }}
{% endblock %}

{% block content %}
<!-- Hero Section -->
<section class="hero">
  <div class="hero__inner">
    <div class="hero__badge">Open Source Framework</div>
    <h1 class="hero__title">siirl-agentic</h1>
    <p class="hero__subtitle">
      Asynchronous Multi-Turn <strong>Agentic</strong> Reinforcement Learning Framework
    </p>
    <p class="hero__desc">
      Purpose-built for multi-turn agent trajectories with tool interaction.
      Natively supports PPO/GRPO training on SWE-style tasks where agents call tools,
      receive environment feedback, and accumulate rewards across long interaction horizons.
    </p>
    <div class="hero__actions">
      <a href="get-started/quickstart/" class="md-button md-button--primary">
        Get Started &rarr;
      </a>
      <a href="https://github.com/sii-research/siirl-agentic" class="md-button">
        GitHub &rarr;
      </a>
    </div>
  </div>
</section>

<!-- Feature Cards (4 核心特性，对标 ROLL 的 Core Advantages) -->
<section class="features">
  <div class="features__grid">

    <div class="feature-card">
      <div class="feature-card__icon">&#x1F916;</div>
      <h3>Native Agentic Trajectory Training</h3>
      <p>Token-level log-probabilities and per-turn loss masks tracked across entire
         multi-turn trajectories. No wrapper hacks, no offline collection.</p>
    </div>

    <div class="feature-card">
      <div class="feature-card__icon">&#x1F50C;</div>
      <h3>Pluggable AgentFlow Protocol</h3>
      <p>Three-stage contract — preprocess, generate, reward — loaded at runtime from YAML.
         Switch tasks by editing one config file.</p>
    </div>

    <div class="feature-card">
      <div class="feature-card__icon">&#x26A1;</div>
      <h3>MPMD Async Execution Engine</h3>
      <p>Trainer and RolloutManager run as independent Ray actors. Generation and
         optimization overlap, eliminating pipeline stalls.</p>
    </div>

    <div class="feature-card">
      <div class="feature-card__icon">&#x1F527;</div>
      <h3>AIO: Elastic Tool Infrastructure</h3>
      <p>Three-tier distributed scheduler with Holt-Winters auto-scaling.
         Handles 1000+ concurrent tool calls without cascading timeouts.</p>
    </div>

  </div>
</section>

<!-- Architecture Diagram -->
<section class="architecture">
  <h2>Architecture</h2>
  <pre class="mermaid">
graph TB
    subgraph MainRunner["MainRunner (Ray Actor)"]
        DC[DataCoordinator<br/>Dataloader + DataBuffer]
        RM[RolloutManager<br/>SGLang + AgentFlow + ToolEnv/AIO]
        TG[TrainerGroup<br/>Actor + Ref + Critic]
    end
    DC -->|batch| RM
    RM -->|rollout data| DC
    TG -->|param sync| RM
    DC -->|training batch| TG
  </pre>
</section>

<!-- Comparison Table -->
<section class="comparison">
  <h2>Why Agentic RL Needs a Dedicated Framework</h2>
  <table>
    <thead>
      <tr><th>Dimension</th><th>Single-Turn RL</th><th>Agentic RL (siirl-agentic)</th></tr>
    </thead>
    <tbody>
      <tr><td>Interaction</td><td>One prompt → one response</td><td>Multi-turn: generate → tool → env → generate ...</td></tr>
      <tr><td>Trajectory length</td><td>Fixed, predictable</td><td>Variable, 10-50+ turns</td></tr>
      <tr><td>Latency profile</td><td>Uniform (GPU-bound)</td><td>Heterogeneous (tool calls: 100ms - 30s)</td></tr>
      <tr><td>Reward signal</td><td>End-of-sequence scalar</td><td>Per-turn environment + final outcome</td></tr>
      <tr><td>Tool management</td><td>N/A</td><td>Distributed AIO scheduler, auto-scaling</td></tr>
    </tbody>
  </table>
</section>

<!-- Quick Start -->
<section class="quickstart">
  <h2>Quick Start</h2>
  <div class="highlight">
    <pre><code class="language-bash">git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic && pip install -e ".[gpu]"

# GRPO training: 4 GPUs training + 4 GPUs rollout
bash examples/grpo_train/run_qwen3_8b_separated.sh

# Agentic training with tool interaction
bash examples/AIO/run_qwen3_8b.sh</code></pre>
  </div>
</section>
{% endblock %}
```

### 5.2 自定义样式（overrides/stylesheets/extra.css）

```css
/* ===== Hero Section ===== */
.hero {
  padding: 4rem 1rem 3rem;
  text-align: center;
  background: linear-gradient(135deg,
    var(--md-primary-fg-color--dark) 0%,
    var(--md-default-bg-color) 100%);
}
.hero__badge {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--md-primary-fg-color);
  border: 1px solid var(--md-primary-fg-color);
  border-radius: 999px;
  margin-bottom: 1rem;
}
.hero__title {
  font-size: 3rem;
  font-weight: 800;
  margin: 0.5rem 0;
}
.hero__subtitle {
  font-size: 1.25rem;
  opacity: 0.85;
  max-width: 640px;
  margin: 0 auto 1rem;
}
.hero__desc {
  max-width: 720px;
  margin: 0 auto 2rem;
  opacity: 0.7;
  line-height: 1.7;
}
.hero__actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}

/* ===== Feature Cards ===== */
.features {
  padding: 3rem 1rem;
  max-width: 1200px;
  margin: 0 auto;
}
.features__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.5rem;
}
.feature-card {
  padding: 1.5rem;
  border-radius: 12px;
  border: 1px solid var(--md-default-fg-color--lightest);
  transition: transform 0.2s, box-shadow 0.2s;
}
.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.feature-card__icon {
  font-size: 2rem;
  margin-bottom: 0.75rem;
}
.feature-card h3 {
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
}
.feature-card p {
  font-size: 0.9rem;
  opacity: 0.75;
  line-height: 1.6;
}

/* ===== Architecture ===== */
.architecture, .comparison, .quickstart {
  max-width: 900px;
  margin: 2rem auto;
  padding: 0 1rem;
}

/* ===== Comparison Table ===== */
.comparison table th {
  background: var(--md-primary-fg-color);
  color: #fff;
  font-weight: 600;
}
.comparison table tr:nth-child(even) {
  background: var(--md-code-bg-color);
}

/* ===== Responsive ===== */
@media (max-width: 768px) {
  .hero__title { font-size: 2rem; }
  .hero__subtitle { font-size: 1rem; }
  .features__grid { grid-template-columns: 1fr; }
}
```

---

## 六、RST → Markdown 转换

### 6.1 自动化转换脚本

```bash
#!/bin/bash
# scripts/convert-rst-to-md.sh
# 依赖: pandoc >= 3.0

set -euo pipefail

SRC_DIR="$1"    # 例如 en/ 或 zh/
OUT_DIR="$2"    # 例如 docs/ 或 docs_zh/

find "$SRC_DIR" -name "*.rst" | while read -r rst_file; do
    # 保持相对路径，改扩展名
    rel_path="${rst_file#$SRC_DIR/}"
    md_path="$OUT_DIR/${rel_path%.rst}.md"
    mkdir -p "$(dirname "$md_path")"

    # pandoc 转换
    pandoc --from=rst --to=markdown --wrap=none "$rst_file" -o "$md_path"

    # 后处理：修复 Mermaid 块
    # Sphinx: .. mermaid::   →   MkDocs: ```mermaid
    python3 -c "
import re, sys
with open('$md_path', 'r') as f:
    content = f.read()
# 修复 mermaid 块（pandoc 会保留为 raw block）
content = re.sub(
    r'``` \{=rst\}\n\.\. mermaid::\n(.*?)```',
    lambda m: '\`\`\`mermaid\n' + m.group(1).replace('   ', '') + '\`\`\`',
    content, flags=re.DOTALL)
with open('$md_path', 'w') as f:
    f.write(content)
"

    echo "Converted: $rst_file → $md_path"
done
```

### 6.2 手动修复清单

| 需修复项 | 涉及文件数 | 工作量 | 说明 |
|---------|-----------|-------|------|
| Mermaid 块格式 | ~15 | 小 | `.. mermaid::` → ` ```mermaid ` |
| RST 复杂表格 | ~6 | 中 | grid table → pipe table，多行单元格需手动调整 |
| 自定义 admonition | ~20 | 小 | "Who this is for" → `!!! info "Who this is for"` |
| 交叉引用 | ~15 | 中 | `:ref:` / `:doc:` → `[text](path.md)` |
| `.. raw:: html` 块 | 1 | 小 | 仅 root index.rst，迁移后由 home.html 替代 |
| `.. code::` 块 | ~70 | 无 | pandoc 自动转换 |

---

## 七、侧边栏导航对比

### 7.1 当前 Sphinx 侧边栏（扁平）

```
siirl-agentic Documentation     ← 所有条目平铺
  Why Agentic RL?
  Native Agentic Trajectory Training
  Pluggable AgentFlow Protocol
  MPMD Asynchronous Execution Engine
  AIO: Elastic Agentic Tool Infrastructure
  Design Philosophy
  Architecture Overview
  Async Training Lifecycle
  Installation
  Quickstart
  First Agentic Training Job
  Configuration System
  PPO Training
  GRPO Training
  ... (30+ items 一条列表)
```

### 7.2 迁移后 MkDocs Material 侧边栏（分组折叠）

```
Home
▼ Highlights                    ← 可折叠
    Why Agentic RL
    Native Trajectory Training
    AgentFlow Protocol
    MPMD Async Engine
    AIO Tool Infrastructure
▶ Concepts                      ← 折叠状态
▼ Get Started                   ← 当前展开
    Installation
    Quickstart
    First Agentic Job
▶ User Guide                    ← 折叠，含 10 个子页
▶ Advanced
▶ Reference
▶ Developer Guide
  FAQ
```

---

## 八、双语方案

### 现有方案 vs 新方案

| 维度 | Sphinx 现有 | MkDocs Material |
|------|-----------|----------------|
| 实现方式 | `SIIRL_DOC_LANG` 环境变量 + 两次构建 | `mkdocs-static-i18n` 插件，一次构建 |
| 语言切换 | 自定义 JS (144 行 `lang-toggle.js`) | Material 内置 `extra.alternate` 下拉菜单 |
| URL 结构 | `/en/page.html` / `/zh/page.html` | `/en/page/` / `/zh/page/` |
| 维护成本 | 两套 toctree 需手动同步 | 一份 `nav:` 自动应用两个语言 |

迁移后 `lang-toggle.js` 不再需要，语言切换由 Material 主题原生处理。

---

## 九、CI/CD 部署

### GitHub Actions 配置

```yaml
# .github/workflows/docs.yml
name: Deploy Docs
on:
  push:
    branches: [main]
    paths: ['docs/**']
  pull_request:
    paths: ['docs/**']

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r docs/requirements-docs.txt
      - run: cd docs && mkdocs build --strict
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/site
```

---

## 十、预期效果（Before / After）

### 10.1 视觉效果对比

| 维度 | Before (Sphinx) | After (MkDocs Material) |
|------|----------------|------------------------|
| **落地页** | 纯文本 + 2 个内联 HTML 按钮 | Hero 区 + 4 张特性卡片 + CTA 按钮 + 架构图 + 对比表 |
| **侧边栏** | 30+ 项扁平列表 | 8 个可折叠分组，每组 2-10 个子页 |
| **暗色模式** | 基础，配色不统一 | Material Design 3 暗色方案，一键切换 |
| **顶部导航** | 无 | Tab 式顶部导航（Home / Highlights / Get Started / User Guide / Reference）|
| **搜索** | 基础 Sphinx 搜索 | 即时搜索 + 自动建议 + 关键词高亮 |
| **代码块** | 复制按钮 | 复制按钮 + 行号 + 代码注解 + 多 Tab 切换 |
| **移动端** | 基础 | 完善的响应式布局 |
| **面包屑** | 无 | 自动面包屑导航 |
| **上下篇** | 无 | 自动 Previous / Next 按钮 |
| **语言切换** | 自定义 JS 按钮 | 原生下拉菜单 |

### 10.2 功能新增

| 新功能 | 说明 |
|-------|------|
| **内容 Tab** | 安装方式切换（Docker / pip / 源码）|
| **可折叠区块** | `??? note "点击展开"` 折叠详情 |
| **代码注解** | 代码行内添加解释气泡 |
| **社交卡片** | 分享时自动生成 Open Graph 预览图 |
| **键盘快捷键** | `s` 搜索，`/` 聚焦搜索框 |
| **Git 信息** | 每页显示最后修改时间和作者 |

### 10.3 Admonition 样式示例

**Before (Sphinx RST):**
```rst
.. note::

   Who this is for: Users who need custom agentic task logic.
```

**After (MkDocs Material):**
```markdown
!!! info "Who this is for"
    Users who need to define custom agentic task logic without modifying framework internals.

!!! success "What you will get"
    Understanding of the three-stage AgentFlow protocol and how to inject custom logic via config.
```

### 10.4 Tab 内容示例

**After (MkDocs Material):**
```markdown
=== "Docker (Recommended)"
    ```bash
    docker pull sii-research/siirl-agentic:latest
    docker run -dit --gpus all -p 9001:22 --ipc=host --shm-size=10gb ...
    ```

=== "pip"
    ```bash
    pip install -e ".[gpu]"
    ```

=== "Source"
    ```bash
    git clone https://github.com/sii-research/siirl-agentic.git
    cd siirl-agentic && pip install -e ".[dev]"
    ```
```

---

## 十一、实施时间线

| Phase | 内容 | 耗时 | 产出 |
|-------|------|------|------|
| **Phase 1** | MkDocs Material 项目搭建 | 1.5 天 | mkdocs.yml + 目录结构 + overrides |
| **Phase 2** | RST → Markdown 批量转换 | 2 天 | 63 个 MD 文件 + 手动修复 |
| **Phase 3** | 落地页设计实现 | 1.5 天 | home.html + extra.css |
| **Phase 4** | 侧边栏 + 导航调优 | 0.5 天 | nav 结构 + Tab 配置 |
| **Phase 5** | 双语 i18n 配置 | 1 天 | i18n 插件 + 中文导航翻译 |
| **Phase 6** | CI/CD 部署 | 0.5 天 | GitHub Actions workflow |
| **Phase 7** | 内容增强（Tab、Admonition） | 2 天 | 全部 MD 文件样式统一 |
| **总计** | | **~8 个工作日** | |

---

## 十二、风险与回退

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| pandoc 转换破坏复杂表格 | 中 | 低 | 6 个文件手动修复 |
| Mermaid 渲染差异 | 低 | 低 | 同一 mermaid.js 库 |
| i18n 插件限制 | 低 | 中 | 回退到双次构建方案 |
| 落地页不如 ROLL 精致 | 中 | 中 | CSS-only 实现可达 ROLL 80% 效果 |
| 中文搜索质量 | 中 | 低 | 添加 jieba 分词插件 |

**回退方案：** 旧 Sphinx 文件保留在 `_legacy_sphinx/`，如 MkDocs 方案不满意，可随时切回。Markdown 文件也可直接用于 Docusaurus（如未来决定完全对齐 ROLL 技术栈）。
