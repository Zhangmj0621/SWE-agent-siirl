/**
 * Scroll-triggered reveal animations for documentation pages.
 * Uses IntersectionObserver to add `.is-visible` class when elements
 * enter the viewport, triggering CSS animations.
 */
(function () {
  "use strict";

  // Elements to observe for scroll-triggered reveal
  var selectors = [
    ".md-typeset h2",
    ".md-typeset .admonition",
    ".md-typeset details",
    ".md-typeset figure",
    ".md-typeset table:not([class])",
    ".md-typeset .highlight",
    ".md-typeset .tabbed-set",
    ".diagram-hero"
  ];

  function init() {
    if (!("IntersectionObserver" in window)) return;

    // Skip home page — it has its own animations
    if (document.querySelector(".home-hero")) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.08, rootMargin: "0px 0px -40px 0px" }
    );

    var query = selectors.join(", ");
    document.querySelectorAll(query).forEach(function (el) {
      // Skip elements already in viewport on load (above the fold)
      var rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight * 0.85) {
        el.classList.add("is-visible");
        return;
      }
      el.classList.add("scroll-reveal");
      observer.observe(el);
    });
  }

  // MkDocs Material uses instant loading (XHR navigation).
  // Re-initialize on each page navigation.
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () { init(); });
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
