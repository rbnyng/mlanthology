(function () {
  var wrapper = document.querySelector('.year-selector-wrapper');
  if (!wrapper) return;

  var nav = wrapper.querySelector('.year-selector');
  var btnLeft = wrapper.querySelector('.year-selector-arrow-left');
  var btnRight = wrapper.querySelector('.year-selector-arrow-right');
  if (!nav) return;

  var SCROLL_AMOUNT = 200;

  function updateArrows() {
    var scrollLeft = Math.round(nav.scrollLeft);
    var maxScroll = nav.scrollWidth - nav.clientWidth;
    var canLeft = scrollLeft > 1;
    var canRight = scrollLeft < maxScroll - 1;

    if (btnLeft) {
      btnLeft.hidden = !canLeft;
    }
    if (btnRight) {
      btnRight.hidden = !canRight;
    }

    wrapper.classList.toggle('can-scroll-left', canLeft);
    wrapper.classList.toggle('can-scroll-right', canRight);
  }

  // Scroll current year into view on load
  var current = nav.querySelector('.year-selector-current');
  if (current) {
    // Use instant scroll so it's positioned before the user sees it
    current.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'instant' });
  }

  // Arrow click handlers
  if (btnLeft) {
    btnLeft.addEventListener('click', function () {
      nav.scrollBy({ left: -SCROLL_AMOUNT, behavior: 'smooth' });
    });
  }
  if (btnRight) {
    btnRight.addEventListener('click', function () {
      nav.scrollBy({ left: SCROLL_AMOUNT, behavior: 'smooth' });
    });
  }

  // When a timeline item receives keyboard focus, scroll it into view
  var items = nav.querySelectorAll('.year-selector-item');
  for (var i = 0; i < items.length; i++) {
    items[i].addEventListener('focus', function () {
      this.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
    });
  }

  nav.addEventListener('scroll', updateArrows, { passive: true });
  window.addEventListener('resize', updateArrows);

  // Initial check — use requestAnimationFrame to ensure layout is settled
  requestAnimationFrame(function () {
    updateArrows();
  });
})();
