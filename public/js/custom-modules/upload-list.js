(function () {
    function listHtml(files) {
        if (!Array.isArray(files) || !files.length) {
            return '<li class="list-group-item">No uploaded ZIPs found yet.</li>';
        }
        return files.map(function (file) {
            return '<li class="list-group-item d-flex justify-content-between align-items-center"><span>' + file.name + '</span><small>' + file.size + ' bytes</small></li>';
        }).join('');
    }

    window.CustomModuleUploadList = { listHtml: listHtml };
})();