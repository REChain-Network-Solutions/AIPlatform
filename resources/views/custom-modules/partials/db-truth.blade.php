@php($dbTruth = $visibilityReport['db_schema'] ?? [])
<div class="alert alert-light mb-3">
    <strong>Worksuite DB truth:</strong> {{ $dbTruth['detail'] ?? 'No DB truth snapshot available.' }}
    @if(!empty($dbTruth['tables'] ?? []))
        <div class="table-responsive mt-2">
            <table class="table table-sm mb-0">
                <thead><tr><th>Table</th><th>Exists</th><th>Columns</th></tr></thead>
                <tbody>
                @foreach($dbTruth['tables'] as $table => $meta)
                    <tr>
                        <td>{{ $table }}</td>
                        <td>{{ !empty($meta['exists']) ? 'Yes' : 'No' }}</td>
                        <td class="small">{{ implode(', ', array_slice($meta['columns'] ?? [], 0, 8)) }}</td>
                    </tr>
                @endforeach
                </tbody>
            </table>
        </div>
    @endif
</div>
