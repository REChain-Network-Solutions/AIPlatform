{{-- Basic Information Card --}}
<div class="card mb-4">
    <div class="card-header">
        <h5 class="mb-0">{{ __('Basic Information') }}</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="name" class="form-label">{{ __('Project Name') }} <span class="text-danger">*</span></label>
                    <input type="text" class="form-control @error('name') is-invalid @enderror" 
                           id="name" name="name" value="{{ old('name', $project->name ?? '') }}" required>
                    @error('name')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="code" class="form-label">{{ __('Project Code') }}</label>
                    <input type="text" class="form-control @error('code') is-invalid @enderror" 
                           id="code" name="code" value="{{ old('code', $project->code ?? '') }}">
                    @error('code')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12">
                <div class="mb-3">
                    <label for="description" class="form-label">{{ __('Description') }}</label>
                    <textarea class="form-control @error('description') is-invalid @enderror" 
                              id="description" name="description" rows="3">{{ old('description', $project->description ?? '') }}</textarea>
                    @error('description')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
    </div>
</div>

{{-- Project Details Card --}}
<div class="card mb-4">
    <div class="card-header">
        <h5 class="mb-0">{{ __('Project Details') }}</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="client_id" class="form-label">{{ __('Client') }}</label>
                    <select class="form-select select2-ajax @error('client_id') is-invalid @enderror" 
                            id="client_id" name="client_id" 
                            data-placeholder="{{ __('Select Client') }}"
                            data-ajax--url="{{ route('companies.selectSearch') }}">
                        <option value="">{{ __('Select Client') }}</option>
                        @if(isset($project) && $project->client)
                            <option value="{{ $project->client->id }}" selected>{{ $project->client->name }}</option>
                        @endif
                    </select>
                    @error('client_id')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="project_manager_id" class="form-label">{{ __('Project Manager') }}</label>
                    <select class="form-select select2-ajax @error('project_manager_id') is-invalid @enderror" 
                            id="project_manager_id" name="project_manager_id"
                            data-placeholder="{{ __('Select Project Manager') }}"
                            data-ajax--url="{{ route('users.selectSearch') }}">
                        <option value="">{{ __('Select Project Manager') }}</option>
                        @if(isset($project) && $project->projectManager)
                            <option value="{{ $project->projectManager->id }}" selected>{{ $project->projectManager->getFullName() }}</option>
                        @endif
                    </select>
                    @error('project_manager_id')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12 col-md-4">
                <div class="mb-3">
                    <label for="type" class="form-label">{{ __('Type') }} <span class="text-danger">*</span></label>
                    <select class="form-select @error('type') is-invalid @enderror" id="type" name="type" required>
                        @foreach(\Modules\PMCore\app\Enums\ProjectType::cases() as $type)
                            <option value="{{ $type->value }}" {{ old('type', isset($project) ? $project->type->value : 'client') == $type->value ? 'selected' : '' }}>
                                {{ $type->label() }}
                            </option>
                        @endforeach
                    </select>
                    @error('type')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-4">
                <div class="mb-3">
                    <label for="status" class="form-label">{{ __('Status') }} <span class="text-danger">*</span></label>
                    <select class="form-select @error('status') is-invalid @enderror" id="status" name="status" required>
                        @foreach(\Modules\PMCore\app\Enums\ProjectStatus::cases() as $status)
                            <option value="{{ $status->value }}" {{ old('status', isset($project) ? $project->status->value : 'planning') == $status->value ? 'selected' : '' }}>
                                {{ $status->label() }}
                            </option>
                        @endforeach
                    </select>
                    @error('status')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-4">
                <div class="mb-3">
                    <label for="priority" class="form-label">{{ __('Priority') }} <span class="text-danger">*</span></label>
                    <select class="form-select @error('priority') is-invalid @enderror" id="priority" name="priority" required>
                        @foreach(\Modules\PMCore\app\Enums\ProjectPriority::cases() as $priority)
                            <option value="{{ $priority->value }}" {{ old('priority', isset($project) ? $project->priority->value : 'medium') == $priority->value ? 'selected' : '' }}>
                                {{ $priority->label() }}
                            </option>
                        @endforeach
                    </select>
                    @error('priority')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
    </div>
</div>

{{-- Schedule & Budget Card --}}
<div class="card mb-4">
    <div class="card-header">
        <h5 class="mb-0">{{ __('Schedule & Budget') }}</h5>
    </div>
    <div class="card-body">
        <div class="row">
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="start_date" class="form-label">{{ __('Start Date') }}</label>
                    <input type="text" class="form-control flatpickr @error('start_date') is-invalid @enderror" 
                           id="start_date" name="start_date" value="{{ old('start_date', isset($project) && $project->start_date ? $project->start_date->format('Y-m-d') : '') }}" 
                           placeholder="{{ __('Select start date') }}">
                    @error('start_date')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="end_date" class="form-label">{{ __('End Date') }}</label>
                    <input type="text" class="form-control flatpickr @error('end_date') is-invalid @enderror" 
                           id="end_date" name="end_date" value="{{ old('end_date', isset($project) && $project->end_date ? $project->end_date->format('Y-m-d') : '') }}" 
                           placeholder="{{ __('Select end date') }}">
                    @error('end_date')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="budget" class="form-label">{{ __('Budget') }}</label>
                    <input type="number" class="form-control @error('budget') is-invalid @enderror" 
                           id="budget" name="budget" value="{{ old('budget', $project->budget ?? '') }}" 
                           step="0.01" min="0">
                    @error('budget')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="hourly_rate" class="form-label">{{ __('Hourly Rate') }}</label>
                    <input type="number" class="form-control @error('hourly_rate') is-invalid @enderror" 
                           id="hourly_rate" name="hourly_rate" value="{{ old('hourly_rate', $project->hourly_rate ?? '') }}" 
                           step="0.01" min="0">
                    @error('hourly_rate')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
        <div class="row">
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <label for="color_code" class="form-label">{{ __('Project Color') }}</label>
                    <input type="color" class="form-control @error('color_code') is-invalid @enderror" 
                           id="color_code" name="color_code" value="{{ old('color_code', $project->color_code ?? '#007bff') }}">
                    @error('color_code')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
            <div class="col-12 col-md-6">
                <div class="mb-3">
                    <div class="form-check form-switch mt-4">
                        <input class="form-check-input @error('is_billable') is-invalid @enderror" 
                               type="checkbox" id="is_billable" name="is_billable" value="1" 
                               {{ old('is_billable', $project->is_billable ?? true) ? 'checked' : '' }}>
                        <label class="form-check-label" for="is_billable">
                            {{ __('Is Billable') }}
                        </label>
                    </div>
                    @error('is_billable')
                        <div class="invalid-feedback">{{ $message }}</div>
                    @enderror
                </div>
            </div>
        </div>
    </div>
</div>
