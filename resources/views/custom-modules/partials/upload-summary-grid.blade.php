<div class="col-md-12 mb-3">
    <div class="row">
        <div class="col-md-4 mb-3">
            <div class="card border h-100"><div class="card-body"><div class="text-muted">Stored ZIPs</div><div class="h4 mb-0">{{ count($moduleFiles ?? []) }}</div></div></div>
        </div>
        <div class="col-md-4 mb-3">
            <div class="card border h-100"><div class="card-body"><div class="text-muted">Upload path</div><div class="small text-break">{{ $updateFilePath }}</div></div></div>
        </div>
        <div class="col-md-4 mb-3">
            <div class="card border h-100"><div class="card-body"><div class="text-muted">Analyzer mode</div><div class="small">Ready for Analyze ZIP / Safe Fix / Export JSON</div></div></div>
        </div>
    </div>
</div>
@includeIf('custom-modules.partials.history')
@includeIf('custom-modules.partials.doctor-panels')
