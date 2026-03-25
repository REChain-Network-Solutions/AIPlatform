@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', __('Tasks'))
@section('titlebar_title', __('Tasks'))
@section('titlebar_subtitle', __('Manage your tasks across all projects and records.'))

@section('titlebar_actions')
    {{-- View toggle --}}
    <div class="flex items-center gap-1 rounded-lg border border-border bg-muted p-0.5">
        <a href="{{ route('dashboard.tasks.index', array_merge(request()->query(), ['view' => 'kanban'])) }}"
            @class(['flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition', 'bg-background shadow-xs text-heading-foreground' => $view === 'kanban', 'text-muted-foreground hover:text-heading-foreground' => $view !== 'kanban'])>
            <x-tabler-layout-kanban class="w-4" /> {{ __('Board') }}
        </a>
        <a href="{{ route('dashboard.tasks.index', array_merge(request()->query(), ['view' => 'list'])) }}"
            @class(['flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition', 'bg-background shadow-xs text-heading-foreground' => $view === 'list', 'text-muted-foreground hover:text-heading-foreground' => $view !== 'list'])>
            <x-tabler-list class="w-4" /> {{ __('List') }}
        </a>
    </div>

    {{-- Voice input button --}}
    <button
        type="button"
        id="voice-task-btn"
        class="flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-sm text-muted-foreground shadow-xs transition hover:text-primary"
        title="{{ __('Create task by voice') }}"
    >
        <x-tabler-microphone class="w-4" />
        {{ __('Voice') }}
    </button>

    <x-button href="{{ route('dashboard.tasks.create') }}" size="sm" variant="primary">
        <x-tabler-plus class="w-4" /> {{ __('New Task') }}
    </x-button>
@endsection

@section('content')
<div class="py-10">

    {{-- ── Voice / AI input panel ────────────────────────────────────────── --}}
    <div id="voice-panel" class="mb-6 hidden rounded-2xl border border-primary/30 bg-primary/5 p-4">
        <div class="flex items-start gap-4">
            <div class="flex-1">
                <p class="mb-1 text-sm font-medium text-heading-foreground">{{ __('Speak or type a task') }}</p>
                <p class="text-xs text-muted-foreground">{{ __('e.g. "Call John tomorrow at 2pm, high priority" or "Review proposal by Friday"') }}</p>
                <div class="mt-3 flex items-center gap-2">
                    <input
                        type="text"
                        id="voice-input-text"
                        class="lqd-input lqd-input-sm flex-1"
                        placeholder="{{ __('Type or voice input will appear here…') }}"
                    />
                    <button type="button" id="voice-record-btn"
                        class="flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white transition hover:bg-primary/90">
                        <x-tabler-microphone class="w-4" id="voice-icon" />
                        <span id="voice-btn-label">{{ __('Record') }}</span>
                    </button>
                    <button type="button" id="voice-parse-btn"
                        class="flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-sm hover:bg-accent">
                        <x-tabler-wand class="w-4" /> {{ __('Parse with AI') }}
                    </button>
                    <button type="button" id="voice-panel-close" class="text-muted-foreground hover:text-heading-foreground">
                        <x-tabler-x class="w-4" />
                    </button>
                </div>
                {{-- Parsed preview --}}
                <div id="voice-parsed-preview" class="mt-3 hidden rounded-xl border border-border bg-background p-3">
                    <p class="mb-2 text-xs font-medium text-muted-foreground">{{ __('AI extracted:') }}</p>
                    <div id="voice-parsed-content" class="text-sm text-heading-foreground"></div>
                    <div class="mt-3 flex gap-2">
                        <button type="button" id="voice-create-task-btn"
                            class="rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primary/90">
                            {{ __('Create Task') }}
                        </button>
                        <button type="button" id="voice-edit-first-btn"
                            class="rounded-lg border border-border px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent">
                            {{ __('Edit First') }}
                        </button>
                    </div>
                </div>
            </div>
            <div id="voice-animation" class="hidden mt-1">
                <div class="flex gap-0.5">
                    @for ($i = 0; $i < 5; $i++)
                        <div class="voice-bar w-1 rounded-full bg-primary" style="height: 8px; animation: voicePulse 0.8s ease-in-out {{ $i * 0.15 }}s infinite alternate;"></div>
                    @endfor
                </div>
            </div>
        </div>
    </div>

    @if ($view === 'kanban')
        {{-- ── KANBAN BOARD ────────────────────────────────────────────────── --}}

        {{-- Filters --}}
        <form method="GET" class="mb-5 flex flex-wrap items-center gap-3">
            <input type="hidden" name="view" value="kanban">
            <select name="priority_id" class="lqd-input lqd-input-md">
                <option value="">{{ __('All priorities') }}</option>
                @foreach ($priorities as $p)
                    <option value="{{ $p->id }}" @selected(request('priority_id') == $p->id)>{{ $p->name }}</option>
                @endforeach
            </select>
            <select name="assignee_id" class="lqd-input lqd-input-md">
                <option value="">{{ __('All assignees') }}</option>
                @foreach ($users as $u)
                    <option value="{{ $u->id }}" @selected(request('assignee_id') == $u->id)>{{ $u->name }}</option>
                @endforeach
            </select>
            <x-button type="submit" size="sm" variant="outline">{{ __('Filter') }}</x-button>
            @if (request()->anyFilled(['priority_id', 'assignee_id']))
                <a href="{{ route('dashboard.tasks.index', ['view' => 'kanban']) }}" class="text-xs text-muted-foreground hover:underline">{{ __('Clear') }}</a>
            @endif
        </form>

        <div class="flex gap-4 overflow-x-auto pb-6" id="tasks-kanban" data-base-url="{{ url('dashboard/tasks') }}">
            @foreach ($statuses as $status)
                <div class="flex w-72 shrink-0 flex-col rounded-2xl border border-border bg-card" data-status-id="{{ $status->id }}">
                    {{-- Column header --}}
                    <div class="flex items-center justify-between border-b border-border px-4 py-3">
                        <div class="flex items-center gap-2">
                            <span class="h-2.5 w-2.5 rounded-full" style="background-color: {{ $status->color }}"></span>
                            <span class="text-sm font-semibold text-heading-foreground">{{ $status->name }}</span>
                            <span class="rounded-full bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">{{ $status->tasks->count() }}</span>
                        </div>
                        @if ($status->is_completed_status)
                            <x-tabler-check class="w-3.5 text-green-500" title="{{ __('Completed status') }}" />
                        @endif
                    </div>

                    {{-- Task cards --}}
                    <div class="tasks-column flex min-h-[160px] flex-col gap-2 p-3" data-status-id="{{ $status->id }}">
                        @foreach ($status->tasks as $task)
                            <div
                                class="task-card group cursor-pointer rounded-xl border border-border bg-background p-3 shadow-xs transition hover:border-primary/40 hover:shadow-sm"
                                data-task-id="{{ $task->id }}"
                                draggable="true"
                            >
                                <div class="flex items-start justify-between gap-2">
                                    {{-- Complete toggle --}}
                                    <button type="button"
                                        class="task-complete-btn mt-0.5 shrink-0 text-muted-foreground hover:text-green-500"
                                        data-task-id="{{ $task->id }}"
                                        title="{{ __('Mark complete') }}"
                                    >
                                        @if ($task->isCompleted())
                                            <x-tabler-circle-check-filled class="w-4 text-green-500" />
                                        @else
                                            <x-tabler-circle class="w-4" />
                                        @endif
                                    </button>
                                    <a href="{{ route('dashboard.tasks.show', $task) }}" class="flex-1 min-w-0">
                                        <span class="block text-sm font-medium leading-snug text-heading-foreground {{ $task->isCompleted() ? 'line-through opacity-60' : '' }}">{{ $task->title }}</span>
                                    </a>
                                </div>

                                <div class="mt-2 flex flex-wrap items-center gap-1.5">
                                    @if ($task->priority)
                                        <span class="rounded-full px-1.5 py-0.5 text-xs font-medium"
                                            style="background-color: {{ $task->priority->color }}22; color: {{ $task->priority->color }}">
                                            {{ $task->priority->name }}
                                        </span>
                                    @endif
                                    @if ($task->due_date)
                                        <span @class(['flex items-center gap-0.5 text-xs', 'text-red-500' => $task->is_overdue, 'text-muted-foreground' => !$task->is_overdue])>
                                            <x-tabler-calendar class="w-3" />
                                            {{ $task->due_date->format('M d') }}
                                        </span>
                                    @endif
                                    @if ($task->subTasks->count() > 0)
                                        <span class="flex items-center gap-0.5 text-xs text-muted-foreground">
                                            <x-tabler-subtask class="w-3" /> {{ $task->subTasks->count() }}
                                        </span>
                                    @endif
                                    @if ($task->assignedTo)
                                        <span class="ms-auto flex h-5 w-5 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary" title="{{ $task->assignedTo->name }}">
                                            {{ strtoupper(substr($task->assignedTo->name, 0, 1)) }}
                                        </span>
                                    @endif
                                </div>
                            </div>
                        @endforeach
                    </div>

                    {{-- Quick add --}}
                    <div class="border-t border-border px-3 py-2">
                        <a href="{{ route('dashboard.tasks.create', ['status_id' => $status->id]) }}"
                            class="flex w-full items-center gap-1 rounded-lg px-2 py-1.5 text-xs text-muted-foreground hover:bg-muted hover:text-heading-foreground">
                            <x-tabler-plus class="w-3.5" /> {{ __('Add task') }}
                        </a>
                    </div>
                </div>
            @endforeach
        </div>

    @else
        {{-- ── LIST VIEW ─────────────────────────────────────────────────── --}}

        <form method="GET" class="mb-6 flex flex-wrap items-center gap-3">
            <input type="hidden" name="view" value="list">
            <x-form.input name="search" placeholder="{{ __('Search tasks…') }}" value="{{ request('search') }}" class="w-56" />
            <select name="status_id" class="lqd-input lqd-input-md">
                <option value="">{{ __('All statuses') }}</option>
                @foreach ($statuses as $s)
                    <option value="{{ $s->id }}" @selected(request('status_id') == $s->id)>{{ $s->name }}</option>
                @endforeach
            </select>
            <select name="priority_id" class="lqd-input lqd-input-md">
                <option value="">{{ __('All priorities') }}</option>
                @foreach ($priorities as $p)
                    <option value="{{ $p->id }}" @selected(request('priority_id') == $p->id)>{{ $p->name }}</option>
                @endforeach
            </select>
            <label class="flex items-center gap-1.5 text-sm text-muted-foreground cursor-pointer">
                <input type="checkbox" name="overdue" value="1" class="lqd-checkbox" {{ request('overdue') ? 'checked' : '' }}>
                {{ __('Overdue only') }}
            </label>
            <x-button type="submit" size="sm" variant="outline">{{ __('Filter') }}</x-button>
            @if (request()->anyFilled(['search','status_id','priority_id','overdue']))
                <a href="{{ route('dashboard.tasks.index', ['view' => 'list']) }}" class="text-xs text-muted-foreground hover:underline">{{ __('Clear') }}</a>
            @endif
        </form>

        <div class="overflow-hidden rounded-2xl border border-border bg-card shadow-xs">
            <table class="w-full text-sm">
                <thead class="border-b border-border bg-muted/40 text-left text-xs font-medium text-muted-foreground">
                    <tr>
                        <th class="w-6 px-4 py-3"></th>
                        <th class="px-4 py-3">{{ __('Task') }}</th>
                        <th class="px-4 py-3">{{ __('Status') }}</th>
                        <th class="px-4 py-3">{{ __('Priority') }}</th>
                        <th class="px-4 py-3">{{ __('Assigned') }}</th>
                        <th class="px-4 py-3">{{ __('Due') }}</th>
                        <th class="px-4 py-3 text-right">{{ __('Actions') }}</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-border">
                    @forelse ($tasks as $task)
                        <tr class="hover:bg-accent/20 {{ $task->isCompleted() ? 'opacity-60' : '' }}">
                            <td class="px-4 py-3">
                                <button type="button" class="task-complete-btn text-muted-foreground hover:text-green-500" data-task-id="{{ $task->id }}">
                                    @if ($task->isCompleted())
                                        <x-tabler-circle-check-filled class="w-4 text-green-500" />
                                    @else
                                        <x-tabler-circle class="w-4" />
                                    @endif
                                </button>
                            </td>
                            <td class="px-4 py-3">
                                <a href="{{ route('dashboard.tasks.show', $task) }}"
                                    class="font-medium text-heading-foreground hover:text-primary {{ $task->isCompleted() ? 'line-through' : '' }}">
                                    {{ $task->title }}
                                </a>
                                @if ($task->subTasks->count() > 0)
                                    <span class="ms-1 text-xs text-muted-foreground">· {{ $task->subTasks->count() }} sub-tasks</span>
                                @endif
                            </td>
                            <td class="px-4 py-3">
                                @if ($task->status)
                                    <span class="rounded-full px-2 py-0.5 text-xs"
                                        style="background-color: {{ $task->status->color }}22; color: {{ $task->status->color }}">
                                        {{ $task->status->name }}
                                    </span>
                                @else
                                    <span class="text-muted-foreground">—</span>
                                @endif
                            </td>
                            <td class="px-4 py-3">
                                @if ($task->priority)
                                    <span class="rounded-full px-2 py-0.5 text-xs"
                                        style="background-color: {{ $task->priority->color }}22; color: {{ $task->priority->color }}">
                                        {{ $task->priority->name }}
                                    </span>
                                @else
                                    <span class="text-muted-foreground">—</span>
                                @endif
                            </td>
                            <td class="px-4 py-3 text-xs text-muted-foreground">{{ $task->assignedTo?->name ?? '—' }}</td>
                            <td class="px-4 py-3 text-xs {{ $task->is_overdue ? 'font-semibold text-red-500' : 'text-muted-foreground' }}">
                                {{ $task->due_date?->format('M d, Y') ?? '—' }}
                            </td>
                            <td class="px-4 py-3 text-right">
                                <div class="flex items-center justify-end gap-2">
                                    <a href="{{ route('dashboard.tasks.edit', $task) }}" class="text-muted-foreground hover:text-primary">
                                        <x-tabler-pencil class="w-4" />
                                    </a>
                                    <form method="POST" action="{{ route('dashboard.tasks.destroy', $task) }}" onsubmit="return confirm('{{ __('Delete this task?') }}')">
                                        @csrf @method('DELETE')
                                        <button type="submit" class="text-muted-foreground hover:text-red-500"><x-tabler-trash class="w-4" /></button>
                                    </form>
                                </div>
                            </td>
                        </tr>
                    @empty
                        <tr>
                            <td colspan="7" class="py-16 text-center text-muted-foreground">
                                <x-tabler-checkbox class="mx-auto mb-3 w-10 opacity-30" />
                                <p>{{ __('No tasks found.') }}</p>
                                <a href="{{ route('dashboard.tasks.create') }}" class="mt-2 inline-flex items-center gap-1 text-sm text-primary hover:underline">
                                    <x-tabler-plus class="w-4" /> {{ __('Create a task') }}
                                </a>
                            </td>
                        </tr>
                    @endforelse
                </tbody>
            </table>
        </div>
        <div class="mt-4">{{ $tasks->links() }}</div>
    @endif

</div>
@endsection

@push('style')
<style>
@keyframes voicePulse {
    from { height: 4px; }
    to   { height: 20px; }
}
</style>
@endpush

@push('script')
<script>
(function () {
    const CSRF = document.querySelector('meta[name="csrf-token"]')?.content;
    const baseUrl = '{{ url('dashboard/tasks') }}';

    // ── Voice panel toggle ────────────────────────────────────────────────
    document.getElementById('voice-task-btn')?.addEventListener('click', () => {
        document.getElementById('voice-panel').classList.toggle('hidden');
    });
    document.getElementById('voice-panel-close')?.addEventListener('click', () => {
        document.getElementById('voice-panel').classList.add('hidden');
    });

    // ── Web Speech API ────────────────────────────────────────────────────
    let recognition = null;
    let isRecording = false;
    const recordBtn = document.getElementById('voice-record-btn');
    const voiceIcon = document.getElementById('voice-icon');
    const voiceLabel = document.getElementById('voice-btn-label');
    const voiceAnim = document.getElementById('voice-animation');
    const voiceInput = document.getElementById('voice-input-text');

    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SR();
        recognition.continuous = false;
        recognition.lang = document.documentElement.lang || 'en-US';
        recognition.interimResults = true;

        recognition.onresult = (e) => {
            const transcript = [...e.results].map(r => r[0].transcript).join('');
            voiceInput.value = transcript;
        };
        recognition.onend = () => {
            isRecording = false;
            recordBtn.classList.remove('bg-red-500');
            recordBtn.classList.add('bg-primary');
            voiceLabel.textContent = '{{ __('Record') }}';
            voiceAnim.classList.add('hidden');
        };
    } else {
        recordBtn?.setAttribute('title', '{{ __('Voice input not supported in this browser') }}');
    }

    recordBtn?.addEventListener('click', () => {
        if (!recognition) return;
        if (isRecording) {
            recognition.stop();
        } else {
            voiceInput.value = '';
            recognition.start();
            isRecording = true;
            recordBtn.classList.remove('bg-primary');
            recordBtn.classList.add('bg-red-500');
            voiceLabel.textContent = '{{ __('Stop') }}';
            voiceAnim.classList.remove('hidden');
        }
    });

    // ── AI Parse ─────────────────────────────────────────────────────────
    let parsedData = {};
    document.getElementById('voice-parse-btn')?.addEventListener('click', async () => {
        const input = voiceInput.value.trim();
        if (!input) return;

        const btn = document.getElementById('voice-parse-btn');
        btn.textContent = '{{ __('Parsing…') }}';
        btn.disabled = true;

        try {
            const res = await fetch(`${baseUrl}/ai-parse`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-TOKEN': CSRF },
                body: JSON.stringify({ input }),
            });
            const data = await res.json();
            parsedData = data.parsed || {};

            // Show parsed preview
            const content = document.getElementById('voice-parsed-content');
            content.innerHTML = '';
            if (parsedData.title)     content.innerHTML += `<div><strong>{{ __('Title') }}:</strong> ${parsedData.title}</div>`;
            if (parsedData.due_date)  content.innerHTML += `<div><strong>{{ __('Due') }}:</strong> ${new Date(parsedData.due_date).toLocaleString()}</div>`;
            if (parsedData.priority_id) content.innerHTML += `<div><strong>{{ __('Priority') }}:</strong> ID ${parsedData.priority_id}</div>`;

            document.getElementById('voice-parsed-preview').classList.remove('hidden');
        } catch (e) {
            alert('{{ __('AI parsing failed. Please try again.') }}');
        } finally {
            btn.textContent = '{{ __('Parse with AI') }}';
            btn.disabled = false;
        }
    });

    // ── AI → Create task directly ─────────────────────────────────────────
    document.getElementById('voice-create-task-btn')?.addEventListener('click', async () => {
        if (!parsedData.title) return;
        const body = {
            ...parsedData,
            voice_transcript: voiceInput.value,
            ai_source: 'voice',
        };
        try {
            const res = await fetch(`${baseUrl}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRF-TOKEN': CSRF, 'Accept': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await res.json();
            if (data.success) {
                window.location.href = `${baseUrl}/${data.task.id}`;
            }
        } catch (e) {
            alert('{{ __('Failed to create task.') }}');
        }
    });

    // ── AI → Edit first ───────────────────────────────────────────────────
    document.getElementById('voice-edit-first-btn')?.addEventListener('click', () => {
        const params = new URLSearchParams(parsedData).toString();
        window.location.href = `${baseUrl}/create?${params}&voice_transcript=${encodeURIComponent(voiceInput.value)}`;
    });

    // ── Kanban drag-drop ──────────────────────────────────────────────────
    const board = document.getElementById('tasks-kanban');
    if (board) {
        let dragging = null;
        board.querySelectorAll('.task-card').forEach(card => {
            card.addEventListener('dragstart', () => { dragging = card; card.classList.add('opacity-50'); });
            card.addEventListener('dragend', () => { dragging = null; card.classList.remove('opacity-50'); });
        });
        board.querySelectorAll('.tasks-column').forEach(col => {
            col.addEventListener('dragover', e => {
                e.preventDefault();
                const after = [...col.querySelectorAll('.task-card:not(.opacity-50)')].reduce((closest, child) => {
                    const box = child.getBoundingClientRect();
                    const offset = e.clientY - box.top - box.height / 2;
                    return offset < 0 && offset > closest.offset ? { offset, element: child } : closest;
                }, { offset: Number.NEGATIVE_INFINITY }).element;
                after ? col.insertBefore(dragging, after) : col.appendChild(dragging);
            });
            col.addEventListener('drop', e => {
                e.preventDefault();
                if (!dragging) return;
                const statusId = col.dataset.statusId;
                const taskId   = dragging.dataset.taskId;
                const order    = [...col.querySelectorAll('.task-card')].indexOf(dragging);
                fetch(`${board.dataset.baseUrl}/${taskId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json', 'X-CSRF-TOKEN': CSRF, 'Accept': 'application/json' },
                    body: JSON.stringify({ task_status_id: statusId, task_order: order }),
                });
            });
        });
    }

    // ── Complete toggle ───────────────────────────────────────────────────
    document.querySelectorAll('.task-complete-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
            e.stopPropagation();
            const taskId = btn.dataset.taskId;
            const res = await fetch(`${baseUrl}/${taskId}/complete`, {
                method: 'PATCH',
                headers: { 'X-CSRF-TOKEN': CSRF, 'Accept': 'application/json' },
            });
            const data = await res.json();
            if (data.success) location.reload();
        });
    });
})();
</script>
@endpush
