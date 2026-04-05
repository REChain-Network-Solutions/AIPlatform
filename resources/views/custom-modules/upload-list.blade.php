<div class="card">
    <div class="card-header">Uploaded modules</div>
    <div class="card-body">
        <div class="row mb-3">
            <div class="col-md-4">
                <x-custom-modules.quick-stat label="Stored ZIPs" :value="(string) (($installerQuickStats['stored_zip_count'] ?? 0))"/>
            </div>
            <div class="col-md-4">
                <x-custom-modules.quick-stat label="Latest upload" :value="(string) (($installerQuickStats['latest_upload'] ?? '—'))"/>
            </div>
            <div class="col-md-4">
                <x-custom-modules.quick-stat label="Total bytes" :value="(string) (($installerQuickStats['total_storage_bytes'] ?? 0))"/>
            </div>
        </div>
        <ul class="list-group" id="custom-module-upload-list">
            @forelse(($moduleFiles ?? []) as $file)
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    <span>{{ $file['name'] }}</span>
                    <small>{{ $file['size'] }} bytes</small>
                </li>
            @empty
                <li class="list-group-item">No uploaded ZIPs found yet.</li>
            @endforelse
        </ul>
    </div>
</div>
