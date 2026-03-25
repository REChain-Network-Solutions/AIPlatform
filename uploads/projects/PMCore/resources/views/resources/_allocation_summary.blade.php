@if($allocations->count() > 0)
    <div class="d-flex flex-column gap-1">
        @foreach($allocations as $allocation)
            <div class="d-flex align-items-center">
                <span class="badge bg-label-primary me-2">{{ $allocation->allocation_percentage }}%</span>
                <small>
                    <a href="{{ route('pmcore.projects.show', $allocation->project_id) }}">
                        {{ $allocation->project->name }}
                    </a>
                    @if($allocation->end_date)
                        <span class="text-muted">({{ $allocation->start_date->format('M d') }} - {{ $allocation->end_date->format('M d') }})</span>
                    @else
                        <span class="text-muted">(from {{ $allocation->start_date->format('M d') }})</span>
                    @endif
                </small>
            </div>
        @endforeach
    </div>
@else
    <span class="text-muted">{{ __('No allocations') }}</span>
@endif