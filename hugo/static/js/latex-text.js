/* latex-text.js — normalize LaTeX text-mode commands for HTML display.
 *
 * Paper titles and abstracts from upstream sources (PMLR, OpenReview,
 * NeurIPS, etc.) contain LaTeX markup.  KaTeX handles math delimiters
 * ($...$, \[...\]) but text-mode commands such as \textsc{}, \texttt{},
 * \emph{} are left as literal backslash text in the DOM.
 *
 * This module transforms those commands into semantic HTML after KaTeX
 * has already rendered.  Pre/code blocks (BibTeX, citation exports) are
 * intentionally skipped — they contain valid LaTeX for copy-paste into
 * .tex files.
 *
 * Execution order: loaded as a regular <script> in <body> (after KaTeX
 * auto-render, which is a deferred script).  DOMContentLoaded handlers
 * fire in registration order, so KaTeX's handler always runs first.
 */
(function () {
  "use strict";

  /* Each entry: [regex, replacer].
   * The content capture group [^{}]* deliberately excludes brace chars so
   * nested commands resolve correctly in successive passes (innermost first).
   * All regexes use the /g flag; lastIndex is reset before each application. */
  var TRANSFORMS = [
    /* \cmd{content} forms ------------------------------------------------- */
    [/\\textsc\{([^{}]*)\}/g,          function (_, c) { return '<span class="lsc">' + c + '</span>'; }],
    [/\\texttt\{([^{}]*)\}/g,          function (_, c) { return '<code>' + c + '</code>'; }],
    [/\\emph\{([^{}]*)\}/g,            function (_, c) { return '<em>' + c + '</em>'; }],
    [/\\textit\{([^{}]*)\}/g,          function (_, c) { return '<em>' + c + '</em>'; }],
    [/\\textbf\{([^{}]*)\}/g,          function (_, c) { return '<strong>' + c + '</strong>'; }],
    [/\\textsuperscript\{([^{}]*)\}/g, function (_, c) { return '<sup>' + c + '</sup>'; }],
    [/\\textsubscript\{([^{}]*)\}/g,   function (_, c) { return '<sub>' + c + '</sub>'; }],
    [/\\url\{([^{}]*)\}/g,             function (_, c) { return '<code>' + c + '</code>'; }],

    /* {\cmd content} forms (old-style LaTeX) ------------------------------- */
    [/\{\\sc\s+([^{}]*?)\}/g,          function (_, c) { return '<span class="lsc">' + c.trimEnd() + '</span>'; }],
    [/\{\\tt\s+([^{}]*?)\}/g,          function (_, c) { return '<code>' + c.trimEnd() + '</code>'; }],
    [/\{\\em\s+([^{}]*?)\}/g,          function (_, c) { return '<em>' + c.trimEnd() + '</em>'; }],
    [/\{\\it\s+([^{}]*?)\}/g,          function (_, c) { return '<em>' + c.trimEnd() + '</em>'; }],
    [/\{\\bf\s+([^{}]*?)\}/g,          function (_, c) { return '<strong>' + c.trimEnd() + '</strong>'; }],

    /* Special characters --------------------------------------------------- */
    /* \& — Hugo HTML-escapes & to &amp; so we match the escaped form.
     * The replacement keeps &amp; (rendered as & in the browser). */
    [/\\&amp;/g,                       function ()      { return '&amp;'; }],

    /* ~ is a non-breaking space in LaTeX.  Any ~ remaining after KaTeX has
     * processed math must be outside a math delimiter, i.e. literal LaTeX. */
    [/~/g,                             function ()      { return '\u00a0'; }],

    /* Dashes — longer pattern first to avoid -- matching inside ---. */
    [/---/g,                           function ()      { return '\u2014'; }],
    [/--/g,                            function ()      { return '\u2013'; }],

    /* Bare braces used for case-protection in BibTeX titles, e.g. {GPT-4}.
     * After all command transforms have run, any remaining {content} that
     * contains neither backslashes nor nested braces is just stripped.
     * KaTeX output never contains literal { chars so this is safe. */
    [/\{([^{}\\]*)\}/g,                function (_, c) { return c; }],
  ];

  function normalizeLatexHtml(html) {
    var s = html;
    /* Iterate until stable so nested commands resolve correctly, e.g.
     * \textsc{\emph{Foo}} needs two passes.  Cap at 5 to be safe. */
    for (var pass = 0; pass < 5; pass++) {
      var prev = s;
      for (var i = 0; i < TRANSFORMS.length; i++) {
        TRANSFORMS[i][0].lastIndex = 0;
        s = s.replace(TRANSFORMS[i][0], TRANSFORMS[i][1]);
      }
      if (s === prev) { break; }
    }
    return s;
  }

  function processElement(el) {
    var before = el.innerHTML;
    var after = normalizeLatexHtml(before);
    if (after !== before) { el.innerHTML = after; }
  }

  document.addEventListener('DOMContentLoaded', function () {
    /* Paper detail page: h1 title and abstract paragraph. */
    var detail = document.querySelectorAll('.paper-detail > h1, .paper-abstract p');
    for (var i = 0; i < detail.length; i++) { processElement(detail[i]); }

    /* Paper list items: venue-year pages and author pages. */
    var titles = document.querySelectorAll('.paper-title');
    for (var j = 0; j < titles.length; j++) { processElement(titles[j]); }
  });
}());
