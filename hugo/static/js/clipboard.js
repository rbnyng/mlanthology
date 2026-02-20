// Copy-to-clipboard for BibTeX
document.addEventListener("DOMContentLoaded", function () {
  // Live region for screen reader announcements
  var liveRegion = document.createElement("span");
  liveRegion.setAttribute("role", "status");
  liveRegion.setAttribute("aria-live", "polite");
  liveRegion.style.position = "absolute";
  liveRegion.style.width = "1px";
  liveRegion.style.height = "1px";
  liveRegion.style.overflow = "hidden";
  liveRegion.style.clip = "rect(0,0,0,0)";
  document.body.appendChild(liveRegion);

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for non-HTTPS or older browsers
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      try {
        var ok = document.execCommand("copy");
        document.body.removeChild(textarea);
        ok ? resolve() : reject(new Error("execCommand copy failed"));
      } catch (e) {
        document.body.removeChild(textarea);
        reject(e);
      }
    });
  }

  function flashCopied(btn, originalLabel) {
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    btn.disabled = true;
    liveRegion.textContent = "";
    liveRegion.textContent = "Copied to clipboard";
    setTimeout(function () {
      btn.textContent = originalLabel;
      btn.classList.remove("copied");
      btn.disabled = false;
    }, 2000);
  }

  function flashError(btn, originalLabel) {
    btn.textContent = "Failed!";
    btn.classList.add("error");
    setTimeout(function () {
      btn.textContent = originalLabel;
      btn.classList.remove("error");
    }, 2000);
  }

  // Citation copy buttons (paper detail pages)
  document.querySelectorAll(".copy-btn[data-clipboard-target]").forEach(function (btn) {
    var originalLabel = btn.textContent;
    btn.addEventListener("click", function () {
      var target = document.querySelector(btn.getAttribute("data-clipboard-target"));
      if (target) {
        copyToClipboard(target.textContent.trim()).then(function () {
          flashCopied(btn, originalLabel);
        }).catch(function () {
          flashError(btn, originalLabel);
        });
      }
    });
  });

  // Inline cite buttons (venue listing pages)
  document.querySelectorAll(".copy-btn-inline[data-bibtex]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      copyToClipboard(btn.getAttribute("data-bibtex")).then(function () {
        flashCopied(btn, "Cite");
      }).catch(function () {
        flashError(btn, "Cite");
      });
    });
  });
});
