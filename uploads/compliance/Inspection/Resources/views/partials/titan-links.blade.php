@php
    /** @var \Modules\Inspection\Services\TitanLinkService $titan */
    $titan = app(\Modules\Inspection\Services\TitanLinkService::class);
@endphp

<div class="card mb-3">
    <div class="card-header d-flex align-items-center justify-content-between">
        <div>
            <h6 class="mb-0">{{ __('inspection::general.titan_panel_title') }}</h6>
            <small class="text-muted">{{ __('inspection::general.titan_panel_help') }}</small>
        </div>

        @php $addInspectionPermission = user()->permission('add_inspection'); @endphp
        @if(in_array($addInspectionPermission, ['all']))
            <a class="btn btn-primary" href="{{ route('inspection_schedules.create') }}">
                {{ __('inspection::general.new_inspection') }}
            </a>
        @endif
    </div>
    <div class="card-body d-flex flex-wrap gap-2">
        <a class="btn btn-outline-primary" href="{{ $titan->docsUrl() }}">
            {{ __('inspection::general.titan_docs_label') }}
        </a>

        <a class="btn btn-outline-secondary" href="{{ $titan->complianceUrl() }}">
            {{ __('inspection::general.titan_compliance_label') }}
        </a>
    </div>
</div>
