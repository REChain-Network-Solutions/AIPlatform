@extends('layouts.layoutMaster')

@section('title', __('Project Status Details'))

@section('content')
<x-breadcrumb
    :title="$projectStatus->name"
    :breadcrumbs="[
        ['name' => __('Home'), 'url' => route('dashboard')],
        ['name' => __('Master Data'), 'url' => route('master-data.index')],
        ['name' => __('Project Management'), 'url' => ''],
        ['name' => __('Project Statuses'), 'url' => route('pmcore.project-statuses.index')],
        ['name' => $projectStatus->name, 'url' => '']
    ]"
    :home-url="route('dashboard')"
/>

<div class="row">
    <div class="col-md-8">
        <!-- Project Status Details -->
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Status Details') }}</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Status Name') }}</label>
                            <p class="mb-0">{{ $projectStatus->name }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Description') }}</label>
                            <p class="mb-0">{{ $projectStatus->description ?: '-' }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Color') }}</label>
                            <p class="mb-0">
                                <span class="badge" style="background-color: {{ $projectStatus->color }}; color: white;">
                                    {{ $projectStatus->color }}
                                </span>
                            </p>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Status') }}</label>
                            <p class="mb-0">
                                <span class="badge bg-label-{{ $projectStatus->is_active ? 'success' : 'secondary' }}">
                                    {{ $projectStatus->is_active ? __('Active') : __('Inactive') }}
                                </span>
                            </p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Default Status') }}</label>
                            <p class="mb-0">
                                <span class="badge bg-label-{{ $projectStatus->is_default ? 'primary' : 'secondary' }}">
                                    {{ $projectStatus->is_default ? __('Yes') : __('No') }}
                                </span>
                            </p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Represents Completion') }}</label>
                            <p class="mb-0">
                                <span class="badge bg-label-{{ $projectStatus->is_completed ? 'success' : 'secondary' }}">
                                    {{ $projectStatus->is_completed ? __('Yes') : __('No') }}
                                </span>
                            </p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">{{ __('Sort Order') }}</label>
                            <p class="mb-0">{{ $projectStatus->sort_order }}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="col-md-4">
        <!-- Usage Statistics -->
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Usage Statistics') }}</h5>
            </div>
            <div class="card-body">
                <div class="d-flex align-items-center mb-3">
                    <div class="avatar">
                        <div class="avatar-initial bg-primary rounded">
                            <i class="bx bx-briefcase text-white"></i>
                        </div>
                    </div>
                    <div class="ms-3">
                        <div class="small mb-1">{{ __('Projects Using This Status') }}</div>
                        <h5 class="card-title mb-0">{{ $projectStatus->projects()->count() }}</h5>
                    </div>
                </div>

                @if($projectStatus->projects()->count() > 0)
                    <div class="mt-3">
                        <h6 class="mb-2">{{ __('Recent Projects') }}</h6>
                        @foreach($projectStatus->projects()->latest()->limit(5)->get() as $project)
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <span class="small">{{ $project->name }}</span>
                                <a href="{{ route('pmcore.projects.show', $project) }}" class="btn btn-sm btn-outline-primary">
                                    {{ __('View') }}
                                </a>
                            </div>
                        @endforeach
                    </div>
                @endif
            </div>
        </div>

        <!-- Actions -->
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">{{ __('Actions') }}</h5>
            </div>
            <div class="card-body">
                <div class="d-grid gap-2">
                    <a href="{{ route('pmcore.project-statuses.index') }}" class="btn btn-outline-secondary">
                        <i class="bx bx-arrow-back me-1"></i>{{ __('Back to List') }}
                    </a>
                    <button class="btn btn-primary" onclick="editStatus({{ $projectStatus->id }})">
                        <i class="bx bx-edit me-1"></i>{{ __('Edit Status') }}
                    </button>
                    @if(!$projectStatus->is_default && $projectStatus->projects()->count() === 0)
                        <button class="btn btn-danger" onclick="deleteStatus({{ $projectStatus->id }})">
                            <i class="bx bx-trash me-1"></i>{{ __('Delete Status') }}
                        </button>
                    @endif
                </div>
            </div>
        </div>
    </div>
</div>

<script>
function editStatus(id) {
    window.location.href = '{{ route("pmcore.project-statuses.index") }}#edit-' + id;
}

function deleteStatus(id) {
    if (confirm('{{ __("Are you sure you want to delete this project status?") }}')) {
        // Add delete functionality here
        console.log('Delete status:', id);
    }
}
</script>

@endsection
