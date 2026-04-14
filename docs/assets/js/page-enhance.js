/* LYNX — page-enhance.js
   Injects a styled hero header on every non-homepage content page.
   Runs after DOMContentLoaded so JTD's own JS has already fired. */
(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    var mainContent = document.querySelector('#main-content, .main-content');
    if (!mainContent) return;

    /* Skip homepage: it has a centered hero div with inline style */
    if (mainContent.querySelector('[style*="text-align: center"]') ||
        mainContent.querySelector('[style*="text-align:center"]')) return;

    var h1 = mainContent.querySelector('h1');
    if (!h1) return;

    /* Build parent label from breadcrumb (last <a> before current page) */
    var parentHTML = '';
    var breadcrumb = document.querySelector('.breadcrumb-nav');
    if (breadcrumb) {
      var links = breadcrumb.querySelectorAll('a');
      if (links.length) {
        parentHTML = '<span class="lynx-hero-label">' +
                     links[links.length - 1].textContent.trim() +
                     '</span>';
      }
    }

    /* Create the hero banner */
    var hero = document.createElement('div');
    hero.className = 'lynx-hero';
    /* Strip anchor-heading SVG that just-the-docs injects into h1 */
    var titleText = h1.childNodes;
    var titleStr = '';
    titleNodes: for (var n = 0; n < titleText.length; n++) {
      var node = titleText[n];
      if (node.nodeType === Node.TEXT_NODE) {
        titleStr += node.textContent;
      } else if (node.nodeName !== 'A' && node.nodeName !== 'SVG' &&
                 !(node.classList && node.classList.contains('anchor-heading'))) {
        titleStr += node.outerHTML || node.textContent;
      }
    }
    titleStr = titleStr.trim() || h1.textContent.trim();

    hero.innerHTML =
      '<div class="lynx-hero-inner">' +
        parentHTML +
        '<h1 class="lynx-hero-title">' + titleStr + '</h1>' +
      '</div>';

    /* Insert before the breadcrumb nav (or before mainContent if no breadcrumb) */
    var wrap = document.querySelector('.main-content-wrap');
    if (wrap) {
      wrap.insertBefore(hero, wrap.firstChild);
    } else {
      mainContent.parentNode.insertBefore(hero, mainContent);
    }

    /* Remove the original h1 from body text */
    h1.parentNode.removeChild(h1);

    /* Number the h2 sections for visual structure */
    var h2s = mainContent.querySelectorAll('h2');
    h2s.forEach(function (h2, i) {
      var marker = document.createElement('span');
      marker.className = 'lynx-section-num';
      marker.textContent = String(i + 1).padStart(2, '0');
      h2.insertBefore(marker, h2.firstChild);
    });
  });
})();
