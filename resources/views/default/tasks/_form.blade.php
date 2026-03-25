@if ($errors->any())
    <div class="mb-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        <ul class="list-inside list-disc space-y-1">
            @foreach ($errors->all() as $error)<li>{{ $error }}</li>@endforeach
        </ul>
    </div>
@endif

{{-- Voice transcript passthrough --}}
@if (request('voice_transcript'))
    <input type="hidden" name="voice_transcript" value="{{ request('voice_transcript') }}">
    <input type="hidden" name="ai_source" value="voice">
@endif

<div class="grid grid-cols-1 gap-5 lg:grid-cols-3">

    {{-- Main column --}}
    <div class="space-y-5 lg:col-span-2">

        <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
            <h3 class="mb-5 text-sm font-semibold text-heading-foreground">{{ __('Task Details') }}</h3>
            <div class="space-y-4">
                <x-form.group label="{{ __('Title') }}" required>
                    <x-form.input
                        name="title"
                        id="task-title"
                        value="{{ old('title', $task?->title ?? request('title')) }}"
                        placeholder="{{ __('What needs to be done?') }}"
                        required
                    />
                </x-form.group>

                <x-form.group label="{{ __('Description') }}">
                    <textarea
                        name="description"
                        rows="4"
                        class="lqd-input lqd-input-md w-full resize-none"
                        placeholder="{{ __('Add any details, context, or acceptance criteria…') }}"
                    >{{ old('description', $task?->description) }}</textarea>
                </x-form.group>

                {{-- Sub-task indicator --}}
                @if (isset($parentTask))
                    <div class="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
                        <x-tabler-subtask class="w-4 shrink-0" />
                        {{ __('Sub-task of:') }} <strong class="text-heading-foreground">{{ $parentTask->title }}</strong>
                        <input type="hidden" name="parent_task_id" value="{{ $parentTask->id }}">
                    </div>
                @endif
            </div>
        </div>

        {{-- Link to a record --}}
        <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
            <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Link to Record') }}</h3>
            <p class="mb-4 text-xs text-muted-foreground">{{ __('Optionally associate this task with a Contact, Lead, Deal, or Company.') }}</p>
            <div class="grid grid-cols-2 gap-4">
                <x-form.group label="{{ __('Type') }}">
                    <select name="taskable_type" id="taskable-type" class="lqd-input lqd-input-md w-full">
                        <option value="">{{ __('— None —') }}</option>
                        @foreach (['App\\Models\\Crm\\CrmContact' => 'Contact', 'App\\Models\\Crm\\CrmLead' => 'Lead', 'App\\Models\\Crm\\CrmDeal' => 'Deal', 'App\\Models\\Crm\\CrmCompany' => 'Company'] as $type => $label)
                            <option value="{{ $type }}" @selected(old('taskable_type', $task?->taskable_type) === $type)>{{ $label }}</option>
                        @endforeach
                    </select>
                </x-form.group>
                <x-form.group label="{{ __('Record ID') }}">
                    <x-form.input type="number" name="taskable_id" value="{{ old('taskable_id', $task?->taskable_id) }}" placeholder="{{ __('Record ID') }}" />
                </x-form.group>
            </div>
        </div>

    </div>

    {{-- Sidebar --}}
    <div class="space-y-5">

        <div class="rounded-2xl border border-border bg-card p-5 shadow-xs">
            <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Properties') }}</h3>
            <div class="space-y-4">

                <x-form.group label="{{ __('Status') }}">
                    <select name="task_status_id" class="lqd-input lqd-input-md w-full">
                        <option value="">{{ __('— None —') }}</option>
                        @foreach ($statuses as $s)
                            <option value="{{ $s->id }}" @selected(old('task_status_id', $task?->task_status_id ?? request('status_id')) == $s->id)>{{ $s->name }}</option>
                        @endforeach
                    </select>
                </x-form.group>

                <x-form.group label="{{ __('Priority') }}">
                    <select name="task_priority_id" class="lqd-input lqd-input-md w-full">
                        <option value="">{{ __('— None —') }}</option>
                        @foreach ($priorities as $p)
                            <option value="{{ $p->id }}"
                                @selected(old('task_priority_id', $task?->task_priority_id ?? request('priority_id')) == $p->id)
                                style="color: {{ $p->color }}">
                                {{ $p->name }}
                            </option>
                        @endforeach
                    </select>
                </x-form.group>

                <x-form.group label="{{ __('Assign To') }}">
                    <select name="assigned_to_user_id" class="lqd-input lqd-input-md w-full">
                        <option value="">{{ __('— Unassigned —') }}</option>
                        @foreach ($users as $u)
                            <option value="{{ $u->id }}" @selected(old('assigned_to_user_id', $task?->assigned_to_user_id) == $u->id)>{{ $u->name }}</option>
                        @endforeach
                    </select>
                </x-form.group>

                <x-form.group label="{{ __('Due Date') }}">
                    <x-form.input
                        type="datetime-local"
                        name="due_date"
                        value="{{ old('due_date', $task?->due_date?->format('Y-m-d\TH:i') ?? request('due_date')) }}"
                    />
                </x-form.group>

                <x-form.group label="{{ __('Reminder') }}">
                    <x-form.input
                        type="datetime-local"
                        name="reminder_at"
                        value="{{ old('reminder_at', $task?->reminder_at?->format('Y-m-d\TH:i')) }}"
                    />
                </x-form.group>

            </div>
        </div>

        <div class="rounded-2xl border border-border bg-card p-5 shadow-xs">
            <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Time & Effort') }}</h3>
            <div class="grid grid-cols-2 gap-3">
                <x-form.group label="{{ __('Estimated (hrs)') }}">
                    <x-form.input type="number" name="estimated_hours" value="{{ old('estimated_hours', $task?->estimated_hours) }}" min="0" step="0.5" />
                </x-form.group>
                <x-form.group label="{{ __('Actual (hrs)') }}">
                    <x-form.input type="number" name="actual_hours" value="{{ old('actual_hours', $task?->actual_hours) }}" min="0" step="0.5" />
                </x-form.group>
            </div>
            <div class="mt-3">
                <label class="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
                    <input type="checkbox" name="is_milestone" value="1" class="lqd-checkbox"
                        {{ old('is_milestone', $task?->is_milestone) ? 'checked' : '' }}>
                    {{ __('Mark as milestone') }}
                </label>
            </div>
        </div>

    </div>

</div>
