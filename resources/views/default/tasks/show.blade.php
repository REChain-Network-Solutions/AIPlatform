@extends('panel.layout.app', ['disable_tblr' => true])

@section('title', $task->title)
@section('titlebar_title', $task->title)
@section('titlebar_subtitle', $task->status?->name ?? __('No status'))

@section('titlebar_actions')
    <button type="button" id="complete-btn"
        class="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition {{ $task->isCompleted() ? 'bg-green-100 text-green-700 hover:bg-green-200' : 'border border-border bg-background text-muted-foreground hover:text-heading-foreground' }}"
        data-task-id="{{ $task->id }}"
    >
        @if ($task->isCompleted())
            <x-tabler-circle-check-filled class="w-4" /> {{ __('Completed') }}
        @else
            <x-tabler-circle class="w-4" /> {{ __('Mark Complete') }}
        @endif
    </button>
    <x-button href="{{ route('dashboard.tasks.edit', $task) }}" size="sm" variant="outline">
        <x-tabler-pencil class="w-4" /> {{ __('Edit') }}
    </x-button>
@endsection

@section('content')
<div class="py-10">
    <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">

        {{-- Main --}}
        <div class="space-y-5 lg:col-span-2">

            @if ($task->description)
                <div class="rounded-2xl border border-border bg-card p-6 shadow-xs">
                    <h3 class="mb-3 text-sm font-semibold text-heading-foreground">{{ __('Description') }}</h3>
                    <div class="prose prose-sm max-w-none text-muted-foreground">{{ $task->description }}</div>
                </div>
            @endif

            {{-- Sub-tasks --}}
            <div class="rounded-2xl border border-border bg-card shadow-xs">
                <div class="flex items-center justify-between border-b border-border px-5 py-4">
                    <h3 class="text-sm font-semibold text-heading-foreground">
                        {{ __('Sub-tasks') }}
                        <span class="ms-1 text-xs font-normal text-muted-foreground">({{ $task->subTasks->count() }})</span>
                    </h3>
                    <a href="{{ route('dashboard.tasks.create', ['parent_id' => $task->id]) }}"
                        class="flex items-center gap-1 text-xs text-primary hover:underline">
                        <x-tabler-plus class="w-3.5" /> {{ __('Add') }}
                    </a>
                </div>
                @if ($task->subTasks->isEmpty())
                    <p class="px-5 py-6 text-sm text-muted-foreground">{{ __('No sub-tasks yet.') }}</p>
                @else
                    <ul class="divide-y divide-border">
                        @foreach ($task->subTasks as $sub)
                            <li class="flex items-center gap-3 px-5 py-3">
                                <button type="button" class="task-complete-btn shrink-0 text-muted-foreground hover:text-green-500" data-task-id="{{ $sub->id }}">
                                    @if ($sub->isCompleted())
                                        <x-tabler-circle-check-filled class="w-4 text-green-500" />
                                    @else
                                        <x-tabler-circle class="w-4" />
                                    @endif
                                </button>
                                <a href="{{ route('dashboard.tasks.show', $sub) }}"
                                    class="flex-1 text-sm text-heading-foreground hover:text-primary {{ $sub->isCompleted() ? 'line-through opacity-60' : '' }}">
                                    {{ $sub->title }}
                                </a>
                                @if ($sub->priority)
                                    <span class="shrink-0 rounded-full px-1.5 py-0.5 text-xs"
                                        style="background-color: {{ $sub->priority->color }}22; color: {{ $sub->priority->color }}">
                                        {{ $sub->priority->name }}
                                    </span>
                                @endif
                                @if ($sub->due_date)
                                    <span class="shrink-0 text-xs {{ $sub->is_overdue ? 'text-red-500' : 'text-muted-foreground' }}">
                                        {{ $sub->due_date->format('M d') }}
                                    </span>
                                @endif
                            </li>
                        @endforeach
                    </ul>
                @endif
            </div>

            {{-- Comments --}}
            <div class="rounded-2xl border border-border bg-card shadow-xs">
                <div class="border-b border-border px-5 py-4">
                    <h3 class="text-sm font-semibold text-heading-foreground">{{ __('Comments') }}</h3>
                </div>

                {{-- Comment list --}}
                <div id="comments-list" class="divide-y divide-border">
                    @foreach ($task->comments as $comment)
                        <div class="px-5 py-4">
                            <div class="flex items-start gap-3">
                                <div class="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                                    {{ strtoupper(substr($comment->user->name, 0, 1)) }}
                                </div>
                                <div class="flex-1">
                                    <div class="flex items-center gap-2">
                                        <span class="text-sm font-medium text-heading-foreground">{{ $comment->user->name }}</span>
                                        @if ($comment->is_voice_note)
                                            <span class="flex items-center gap-0.5 rounded-full bg-primary/10 px-1.5 py-0.5 text-xs text-primary">
                                                <x-tabler-microphone class="w-3" /> {{ __('Voice') }}
                                            </span>
                                        @endif
                                        <span class="text-xs text-muted-foreground">{{ $comment->created_at->diffForHumans() }}</span>
                                    </div>
                                    <p class="mt-1 text-sm text-muted-foreground">{{ $comment->content }}</p>
                                </div>
                            </div>
                        </div>
                    @endforeach
                </div>

                {{-- Add comment --}}
                <div class="border-t border-border px-5 py-4">
                    <div class="flex gap-3">
                        <textarea
                            id="comment-input"
                            rows="2"
                            class="lqd-input lqd-input-md flex-1 resize-none"
                            placeholder="{{ __('Add a comment… or click 🎤 to use voice') }}"
                        ></textarea>
                        <div class="flex flex-col gap-2">
                            <button type="button" id="comment-submit" class="rounded-lg bg-primary px-3 py-2 text-sm font-medium text-white hover:bg-primary/90">
                                {{ __('Send') }}
                            </button>
                            <button type="button" id="comment-voice-btn" class="rounded-lg border border-border px-3 py-2 text-sm text-muted-foreground hover:bg-accent" title="{{ __('Voice comment') }}">
                                <x-tabler-microphone class="w-4" />
                            </button>
                        </div>
                    </div>
                </div>
            </div>

        </div>

        {{-- Sidebar --}}
        <div class="space-y-5">

            <div class="rounded-2xl border border-border bg-card p-5 shadow-xs">
                <h3 class="mb-4 text-sm font-semibold text-heading-foreground">{{ __('Details') }}</h3>
                <dl class="space-y-3 text-sm">
                    <div class="flex justify-between">
                        <dt class="text-muted-foreground">{{ __('Status') }}</dt>
                        <dd>
                            @if ($task->status)
                                <span class="rounded-full px-2 py-0.5 text-xs" style="background-color: {{ $task->status->color }}22; color: {{ $task->status->color }}">{{ $task->status->name }}</span>
                            @else —
                            @endif
                        </dd>
                    </div>
                    <div class="flex justify-between">
                        <dt class="text-muted-foreground">{{ __('Priority') }}</dt>
                        <dd>
                            @if ($task->priority)
                                <span class="rounded-full px-2 py-0.5 text-xs" style="background-color: {{ $task->priority->color }}22; color: {{ $task->priority->color }}">{{ $task->priority->name }}</span>
                            @else —
                            @endif
                        </dd>
                    </div>
                    <div class="flex justify-between">
                        <dt class="text-muted-foreground">{{ __('Assigned To') }}</dt>
                        <dd class="font-medium text-heading-foreground">{{ $task->assignedTo?->name ?? '—' }}</dd>
                    </div>
                    <div class="flex justify-between">
                        <dt class="text-muted-foreground">{{ __('Due Date') }}</dt>
                        <dd class="{{ $task->is_overdue ? 'font-semibold text-red-500' : 'text-heading-foreground' }}">
                            {{ $task->due_date?->format('M d, Y H:i') ?? '—' }}
                        </dd>
                    </div>
                    @if ($task->estimated_hours)
                        <div class="flex justify-between">
                            <dt class="text-muted-foreground">{{ __('Estimated') }}</dt>
                            <dd class="text-heading-foreground">{{ $task->estimated_hours }}h</dd>
                        </div>
                    @endif
                    @if ($task->actual_hours)
                        <div class="flex justify-between">
                            <dt class="text-muted-foreground">{{ __('Actual') }}</dt>
                            <dd class="text-heading-foreground">{{ $task->actual_hours }}h</dd>
                        </div>
                    @endif
                    @if ($task->is_milestone)
                        <div class="flex justify-between">
                            <dt class="text-muted-foreground">{{ __('Milestone') }}</dt>
                            <dd><span class="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700">{{ __('Yes') }}</span></dd>
                        </div>
                    @endif
                    @if ($task->isCompleted())
                        <div class="flex justify-between">
                            <dt class="text-muted-foreground">{{ __('Completed') }}</dt>
                            <dd class="text-green-600">{{ $task->completed_at->format('M d, Y') }}</dd>
                        </div>
                    @endif
                </dl>
            </div>

            @if ($task->taskable)
                <div class="rounded-2xl border border-border bg-card p-5 shadow-xs">
                    <h3 class="mb-3 text-sm font-semibold text-heading-foreground">{{ __('Linked Record') }}</h3>
                    <p class="text-xs text-muted-foreground">{{ class_basename($task->taskable_type) }}</p>
                    <p class="mt-1 text-sm font-medium text-heading-foreground">
                        {{ $task->taskable->title ?? $task->taskable->name ?? $task->taskable->full_name ?? "#{$task->taskable_id}" }}
                    </p>
                </div>
            @endif

            @if ($task->ai_metadata)
                <div class="rounded-2xl border border-border bg-card p-5 shadow-xs">
                    <h3 class="mb-3 text-sm font-semibold text-heading-foreground">{{ __('AI / Voice Info') }}</h3>
                    @if (!empty($task->ai_metadata['voice_transcript']))
                        <p class="text-xs text-muted-foreground">{{ __('Transcript:') }}</p>
                        <p class="mt-1 text-xs italic text-heading-foreground">"{{ $task->ai_metadata['voice_transcript'] }}"</p>
                    @endif
                    @if (!empty($task->ai_metadata['source']))
                        <p class="mt-2 text-xs text-muted-foreground">{{ __('Source:') }} <span class="text-heading-foreground">{{ $task->ai_metadata['source'] }}</span></p>
                    @endif
                </div>
            @endif

        </div>

    </div>
</div>
@endsection

@push('script')
<script>
(function () {
    const CSRF   = document.querySelector('meta[name="csrf-token"]')?.content;
    const taskId = {{ $task->id }};
    const baseUrl = '{{ url('dashboard/tasks') }}';

    // Complete toggle
    document.getElementById('complete-btn')?.addEventListener('click', async () => {
        const res  = await fetch(`${baseUrl}/${taskId}/complete`, { method: 'PATCH', headers: { 'X-CSRF-TOKEN': CSRF } });
        const data = await res.json();
        if (data.success) location.reload();
    });

    document.querySelectorAll('.task-complete-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const res = await fetch(`${baseUrl}/${btn.dataset.taskId}/complete`, { method: 'PATCH', headers: { 'X-CSRF-TOKEN': CSRF } });
            if ((await res.json()).success) location.reload();
        });
    });

    // Comment submission
    document.getElementById('comment-submit')?.addEventListener('click', async () => {
        const input = document.getElementById('comment-input');
        const content = input.value.trim();
        if (!content) return;

        const res  = await fetch(`${baseUrl}/${taskId}/comments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-CSRF-TOKEN': CSRF, 'Accept': 'application/json' },
            body: JSON.stringify({ content }),
        });
        const data = await res.json();
        if (data.success) {
            input.value = '';
            // Append comment to list
            const list = document.getElementById('comments-list');
            const c = data.comment;
            list.insertAdjacentHTML('afterbegin', `
                <div class="px-5 py-4">
                    <div class="flex items-start gap-3">
                        <div class="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                            ${c.user.name[0].toUpperCase()}
                        </div>
                        <div class="flex-1">
                            <div class="flex items-center gap-2">
                                <span class="text-sm font-medium text-heading-foreground">${c.user.name}</span>
                                <span class="text-xs text-muted-foreground">${c.created_at}</span>
                            </div>
                            <p class="mt-1 text-sm text-muted-foreground">${c.content}</p>
                        </div>
                    </div>
                </div>
            `);
        }
    });

    // Voice comment
    let voiceRecognition = null;
    const voiceBtn = document.getElementById('comment-voice-btn');
    if (('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) && voiceBtn) {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        voiceRecognition = new SR();
        voiceRecognition.lang = document.documentElement.lang || 'en-US';
        voiceRecognition.onresult = (e) => {
            document.getElementById('comment-input').value = [...e.results].map(r => r[0].transcript).join('');
        };
        voiceRecognition.onend = () => {
            voiceBtn.classList.remove('bg-red-100', 'text-red-500');
            voiceBtn.classList.add('text-muted-foreground');
        };

        let voiceActive = false;
        voiceBtn.addEventListener('click', () => {
            if (voiceActive) {
                voiceRecognition.stop();
                voiceActive = false;
            } else {
                voiceRecognition.start();
                voiceActive = true;
                voiceBtn.classList.add('bg-red-100', 'text-red-500');
                voiceBtn.classList.remove('text-muted-foreground');
            }
        });
    }
})();
</script>
@endpush
