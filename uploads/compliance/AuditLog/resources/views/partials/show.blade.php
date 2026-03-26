<div class="audit-details">
    <!-- Audit Info -->
    <div class="mb-6">
        <h6 class="text-uppercase fw-medium mb-4">{{ __('Audit Information') }}</h6>
        <div class="row g-4">
            <div class="col-6">
                <small class="text-muted d-block mb-1">{{ __('Event') }}</small>
                @php
                    $eventColors = [
                        'created' => 'success',
                        'updated' => 'info',
                        'deleted' => 'danger',
                        'restored' => 'warning',
                    ];
                    $color = $eventColors[$auditLog->event] ?? 'secondary';
                @endphp
                <span class="badge bg-label-{{ $color }}">{{ ucfirst($auditLog->event) }}</span>
            </div>
            <div class="col-6">
                <small class="text-muted d-block mb-1">{{ __('Date & Time') }}</small>
                <span>{{ $auditLog->created_at->format('Y-m-d H:i:s') }}</span>
            </div>
            <div class="col-6">
                <small class="text-muted d-block mb-1">{{ __('Model') }}</small>
                <span class="badge bg-label-primary">{{ class_basename($auditLog->auditable_type) }}</span>
                <span class="text-muted ms-1">#{{ $auditLog->auditable_id }}</span>
            </div>
            <div class="col-6">
                <small class="text-muted d-block mb-1">{{ __('User') }}</small>
                @if($auditLog->user)
                    <div class="d-flex align-items-center">
                        <div class="avatar avatar-sm me-2">
                            <img src="{{ $auditLog->user->avatar_url }}" alt="{{ $auditLog->user->full_name }}" class="rounded-circle">
                        </div>
                        <div>
                            <span class="fw-medium">{{ $auditLog->user->full_name }}</span>
                            <small class="text-muted d-block">{{ $auditLog->user->email }}</small>
                        </div>
                    </div>
                @else
                    <span class="text-muted">{{ __('System') }}</span>
                @endif
            </div>
        </div>
    </div>

    <!-- Technical Details -->
    <div class="mb-6">
        <h6 class="text-uppercase fw-medium mb-4">{{ __('Technical Details') }}</h6>
        <div class="row g-4">
            <div class="col-6">
                <small class="text-muted d-block mb-1">{{ __('IP Address') }}</small>
                <span>{{ $auditLog->ip_address ?: 'N/A' }}</span>
            </div>
            <div class="col-6">
                <small class="text-muted d-block mb-1">{{ __('URL') }}</small>
                <span class="text-truncate d-block" title="{{ $auditLog->url }}">{{ $auditLog->url ?: 'N/A' }}</span>
            </div>
            <div class="col-12">
                <small class="text-muted d-block mb-1">{{ __('User Agent') }}</small>
                <span class="text-truncate d-block" title="{{ $auditLog->user_agent }}">{{ $auditLog->user_agent ?: 'N/A' }}</span>
            </div>
        </div>
    </div>

    <!-- Changes -->
    @if(count($changes) > 0)
        <div>
            <h6 class="text-uppercase fw-medium mb-4">{{ __('Changes') }}</h6>
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>{{ __('Field') }}</th>
                            <th>{{ __('Old Value') }}</th>
                            <th>{{ __('New Value') }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach($changes as $change)
                            <tr>
                                <td class="fw-medium">{{ ucwords(str_replace('_', ' ', $change['field'])) }}</td>
                                <td>
                                    @if($change['old'] !== null)
                                        @if(is_bool($change['old']))
                                            <span class="badge bg-label-{{ $change['old'] ? 'success' : 'danger' }}">
                                                {{ $change['old'] ? __('Yes') : __('No') }}
                                            </span>
                                        @elseif(is_array($change['old']) || is_object($change['old']))
                                            <code>{{ json_encode($change['old']) }}</code>
                                        @else
                                            <span class="text-muted">{{ $change['old'] }}</span>
                                        @endif
                                    @else
                                        <span class="text-muted">-</span>
                                    @endif
                                </td>
                                <td>
                                    @if($change['new'] !== null)
                                        @if(is_bool($change['new']))
                                            <span class="badge bg-label-{{ $change['new'] ? 'success' : 'danger' }}">
                                                {{ $change['new'] ? __('Yes') : __('No') }}
                                            </span>
                                        @elseif(is_array($change['new']) || is_object($change['new']))
                                            <code>{{ json_encode($change['new']) }}</code>
                                        @else
                                            <span class="text-success">{{ $change['new'] }}</span>
                                        @endif
                                    @else
                                        <span class="text-muted">-</span>
                                    @endif
                                </td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>
        </div>
    @else
        <div class="alert alert-info">
            <i class="bx bx-info-circle me-2"></i>
            {{ __('No changes recorded for this event.') }}
        </div>
    @endif
</div>