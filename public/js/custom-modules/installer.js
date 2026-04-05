(function () {
    function refreshAfterUpload(response) {
        if (!response) return;
        if (response.reload === true || (Array.isArray(response.files) && response.files.length)) {
            window.location.reload();
        }
    }

    function mountResults(targetId, html) {
        var el = document.getElementById(targetId);
        if (el) el.innerHTML = html || '';
    }

    function renderCompactSummaryCards(cards) {
        if (!Array.isArray(cards) || !cards.length) return '';
        return cards.map(function (card) {
            var themeClass = card.theme === 'success' ? ' module-doctor-card-alt' : '';
            return '<div class="module-doctor-card' + themeClass + '"><h6>' + card.title + '</h6><p>' + card.body + '</p></div>';
        }).join('');
    }

    window.CustomModuleInstaller = {
        refreshAfterUpload: refreshAfterUpload,
        mountResults: mountResults,
        renderCompactSummaryCards: renderCompactSummaryCards
    };
})();