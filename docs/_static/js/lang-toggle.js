// Language toggle button for siirl-agentic docs (EN <-> ZH)
// Injects a toggle button into the sphinx-book-theme header.
// Supports URL patterns:
//   /en/...  or  /zh/...             (RTD / local direct build)
//   /siirl-agentic/en/...            (GitHub Pages project site)
(function () {
  const STORAGE_KEY = 'siirl-doc-lang';

  function detectCurrent() {
    const { zhIndex } = analyzePath();
    return zhIndex !== -1 ? 'zh' : 'en';
  }

  function otherLang(lang) {
    return lang === 'zh' ? 'en' : 'zh';
  }

  /**
   * Break down the current pathname into repo-root + language segment.
   * Handles:
   *  /en/…                  language as first segment
   *  /siirl-agentic/en/…    GitHub Pages project site (repo root first)
   *  /zh/…  /siirl-agentic/zh/…  (same for Chinese)
   */
  function analyzePath() {
    const parts = window.location.pathname.split('/').filter(Boolean);
    let repoRoot = null;

    // Detect GitHub Pages project-site pattern or explicit repo prefix
    if (
      parts.length > 0 &&
      (window.location.host.endsWith('github.io') || parts[0] === 'siirl-agentic')
    ) {
      repoRoot = parts[0];
    }

    let zhIndex = -1;
    if (parts[0] === 'zh') zhIndex = 0;
    else if (parts[1] === 'zh') zhIndex = 1;

    return { parts, repoRoot, zhIndex };
  }

  function buildTargetUrl(target) {
    const url = new URL(window.location.href);
    const trailingSlash =
      url.pathname.endsWith('/') || url.pathname === '/';
    const { parts, repoRoot, zhIndex } = analyzePath();

    if (target === 'zh') {
      if (zhIndex === -1) {
        if (repoRoot) {
          // /siirl-agentic/page  ->  /siirl-agentic/zh/page
          if (parts.length === 1) parts.push('zh');
          else parts.splice(1, 0, 'zh');
        } else {
          parts.unshift('zh');
        }
      }
    } else {
      // target === 'en': remove zh segment if present
      if (zhIndex !== -1) parts.splice(zhIndex, 1);
    }

    let newPath = '/' + parts.join('/');
    const lastSeg = parts[parts.length - 1] || '';
    if (newPath !== '/' && trailingSlash && !/\.[a-zA-Z0-9]+$/.test(lastSeg)) {
      newPath += '/';
    }
    url.pathname = newPath;
    return url.toString();
  }

  function createButton() {
    const current = detectCurrent();
    const btn = document.createElement('button');
    btn.className = 'btn btn-sm lang-toggle-btn';
    btn.type = 'button';
    btn.setAttribute('data-current', current);
    btn.title =
      current === 'en'
        ? '切换到中文 (当前 EN)'
        : 'Switch to English (当前 中文)';
    btn.innerHTML = `
      <span class="lang-seg" data-lang="en">EN</span>
      <span class="lang-sep">/</span>
      <span class="lang-seg" data-lang="zh">中</span>
    `;
    btn.addEventListener('click', () => {
      const tgt = otherLang(current);
      const targetUrl = buildTargetUrl(tgt);
      try { localStorage.setItem(STORAGE_KEY, tgt); } catch (e) {}
      window.location.href = targetUrl;
    });
    return btn;
  }

  function findContainer() {
    return document.querySelector(
      '.article-header-buttons, .header-article-items__end, ' +
      '.sidebar-header-items, .sidebar-primary-items__end, ' +
      '.bd-header, .bd-sidebar'
    );
  }

  function idealContainer() {
    return document.querySelector('.article-header-buttons');
  }

  function insert(attempt = 0) {
    let c = idealContainer();
    if (!c) c = findContainer();
    if (!c) {
      if (attempt < 40) return setTimeout(() => insert(attempt + 1), 125);
      return;
    }

    // If button already exists but is in the wrong container, move it
    const existing = document.querySelector('.lang-toggle-btn');
    if (existing && c !== existing.parentElement) {
      c.appendChild(existing);
      return;
    }
    if (existing) return;

    const btn = createButton();
    const themeBtn = c.querySelector('.theme-switch-button');
    if (themeBtn && themeBtn.parentElement === c) {
      c.insertBefore(btn, themeBtn);
    } else {
      c.appendChild(btn);
    }
    btn.setAttribute('data-current', detectCurrent());
  }

  document.addEventListener('DOMContentLoaded', () => {
    insert();
    // Watch for dynamic header injection (sphinx-book-theme renders late)
    const obs = new MutationObserver(() => insert());
    obs.observe(document.body, { childList: true, subtree: true });
    setTimeout(() => obs.disconnect(), 5000);
  });
})();
