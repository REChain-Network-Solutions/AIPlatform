<?php

namespace App\Http\Controllers\Tasks;

use App\Http\Controllers\Controller;
use App\Models\Tasks\Task;
use App\Models\Tasks\TaskComment;
use App\Models\Tasks\TaskPriority;
use App\Models\Tasks\TaskStatus;
use App\Models\User;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\View\View;

class TaskController extends Controller
{
    public function index(Request $request): View
    {
        $userId   = Auth::id();
        $statuses = TaskStatus::forUser($userId)->with(['tasks' => function ($q) use ($userId, $request) {
            $q->forUser($userId)->rootTasks()->pending()->with(['priority', 'assignedTo', 'subTasks'])
              ->orderBy('task_order');
            if ($priorityId = $request->input('priority_id')) {
                $q->where('task_priority_id', $priorityId);
            }
            if ($assigneeId = $request->input('assignee_id')) {
                $q->where('assigned_to_user_id', $assigneeId);
            }
        }])->get();

        // Seed defaults if none exist
        if ($statuses->isEmpty()) {
            $this->seedDefaults($userId);
            $statuses = TaskStatus::forUser($userId)->with('tasks')->get();
        }

        $priorities = TaskPriority::forUser($userId)->get();
        $users      = User::where('id', $userId)->orWhereHas('teamMembers', fn ($q) => $q->where('owner_id', $userId))->get(['id', 'name', 'email']);

        // List view query
        $query = Task::forUser($userId)->rootTasks()->with(['status', 'priority', 'assignedTo', 'subTasks'])->latest();

        if ($s = $request->input('search')) {
            $query->where(fn ($q) => $q->where('title', 'like', "%{$s}%")->orWhere('description', 'like', "%{$s}%"));
        }
        if ($statusId = $request->input('status_id')) {
            $query->where('task_status_id', $statusId);
        }
        if ($priorityId = $request->input('priority_id')) {
            $query->where('task_priority_id', $priorityId);
        }
        if ($request->input('overdue')) {
            $query->overdue();
        }

        $tasks = $query->paginate(25)->withQueryString();
        $view  = $request->input('view', 'kanban');

        return view('tasks.index', compact('statuses', 'priorities', 'users', 'tasks', 'view'));
    }

    public function create(Request $request): View
    {
        $userId     = Auth::id();
        $statuses   = TaskStatus::forUser($userId)->get();
        $priorities = TaskPriority::forUser($userId)->get();
        $users      = User::where('id', $userId)->get(['id', 'name', 'email']);
        $parentTask = $request->input('parent_id') ? Task::find($request->input('parent_id')) : null;

        return view('tasks.create', compact('statuses', 'priorities', 'users', 'parentTask'));
    }

    public function store(Request $request): RedirectResponse|JsonResponse
    {
        $validated = $this->validateTask($request);
        $userId    = Auth::id();

        // AI metadata — store voice transcript / AI-extracted fields if present
        $aiMeta = [];
        if ($request->filled('voice_transcript')) {
            $aiMeta['voice_transcript'] = $request->input('voice_transcript');
        }
        if ($request->filled('ai_source')) {
            $aiMeta['source'] = $request->input('ai_source');
        }

        $task = Task::create(array_merge($validated, [
            'user_id'     => $userId,
            'ai_metadata' => !empty($aiMeta) ? $aiMeta : null,
        ]));

        if ($request->expectsJson()) {
            return response()->json(['success' => true, 'task' => $task->load(['status', 'priority', 'assignedTo'])]);
        }

        return redirect()
            ->route('dashboard.tasks.show', $task)
            ->with('success', __('Task created.'));
    }

    public function show(Task $task): View
    {
        $this->authorizeTask($task);
        $task->load(['status', 'priority', 'assignedTo', 'completedBy', 'taskable', 'subTasks.status', 'subTasks.priority', 'parentTask', 'comments.user']);

        $statuses   = TaskStatus::forUser(Auth::id())->get();
        $priorities = TaskPriority::forUser(Auth::id())->get();

        return view('tasks.show', compact('task', 'statuses', 'priorities'));
    }

    public function edit(Task $task): View
    {
        $this->authorizeTask($task);
        $userId     = Auth::id();
        $statuses   = TaskStatus::forUser($userId)->get();
        $priorities = TaskPriority::forUser($userId)->get();
        $users      = User::where('id', $userId)->get(['id', 'name', 'email']);

        return view('tasks.edit', compact('task', 'statuses', 'priorities', 'users'));
    }

    public function update(Request $request, Task $task): RedirectResponse|JsonResponse
    {
        $this->authorizeTask($task);

        // Quick status update (from kanban drag-drop)
        if ($request->has('task_status_id') && $request->keys() === ['task_status_id', 'task_order']) {
            $task->update($request->only(['task_status_id', 'task_order']));

            // Auto-complete if status is a completed status
            if ($task->status?->is_completed_status && !$task->isCompleted()) {
                $task->markCompleted(Auth::id());
            }

            return response()->json(['success' => true]);
        }

        $validated = $this->validateTask($request, $task->id);
        $task->update($validated);

        if ($request->expectsJson()) {
            return response()->json(['success' => true, 'task' => $task->fresh(['status', 'priority', 'assignedTo'])]);
        }

        return redirect()->route('dashboard.tasks.show', $task)->with('success', __('Task updated.'));
    }

    public function destroy(Task $task): RedirectResponse|JsonResponse
    {
        $this->authorizeTask($task);
        $task->delete();

        if (request()->expectsJson()) {
            return response()->json(['success' => true]);
        }

        return redirect()->route('dashboard.tasks.index')->with('success', __('Task deleted.'));
    }

    /**
     * Toggle complete/incomplete.
     */
    public function complete(Task $task): JsonResponse
    {
        $this->authorizeTask($task);

        if ($task->isCompleted()) {
            $task->markIncomplete();
        } else {
            $task->markCompleted(Auth::id());
        }

        return response()->json(['success' => true, 'completed' => $task->fresh()->isCompleted()]);
    }

    /**
     * AI endpoint — parse natural language into a task (voice or text).
     * Expects: { "input": "Remind me to call John tomorrow at 3pm, high priority" }
     * Returns: { "parsed": { title, due_date, priority_id, ... } }
     */
    public function aiParse(Request $request): JsonResponse
    {
        $request->validate(['input' => 'required|string|max:1000']);
        $userId  = Auth::id();
        $input   = $request->input('input');

        // Build context for AI
        $statuses   = TaskStatus::forUser($userId)->get(['id', 'name']);
        $priorities = TaskPriority::forUser($userId)->get(['id', 'name', 'level'])->sortBy('level')->values();

        $systemPrompt = "You are a task extraction assistant. Extract task details from the user's text and return ONLY valid JSON with these keys:\n"
            . "title (string, required), description (string or null), due_date (ISO 8601 datetime or null), "
            . "priority_id (integer or null — match to provided list), assigned_to_name (string or null).\n"
            . "Priority options: " . $priorities->map(fn ($p) => "{$p->id}={$p->name}")->implode(', ') . ".\n"
            . "Return ONLY the JSON object, no explanation.";

        try {
            // Use the platform's AI engine if available
            $aiEngine = app()->bound('ai.engine') ? app('ai.engine') : null;

            if ($aiEngine) {
                $response = $aiEngine->chat([
                    ['role' => 'system', 'content' => $systemPrompt],
                    ['role' => 'user', 'content' => $input],
                ]);
                $parsed = json_decode($response, true) ?? [];
            } else {
                // Fallback: basic regex/keyword extraction without AI
                $parsed = $this->basicNlpParse($input, $priorities);
            }

            return response()->json(['success' => true, 'parsed' => $parsed, 'original' => $input]);
        } catch (\Throwable $e) {
            // Return partial parse so UI can still use it
            return response()->json(['success' => false, 'parsed' => ['title' => $input], 'error' => $e->getMessage()]);
        }
    }

    /**
     * Store a comment on a task.
     */
    public function storeComment(Request $request, Task $task): JsonResponse
    {
        $this->authorizeTask($task);
        $validated = $request->validate([
            'content'       => 'required|string|max:5000',
            'is_voice_note' => 'nullable|boolean',
        ]);

        $comment = TaskComment::create([
            'task_id'       => $task->id,
            'user_id'       => Auth::id(),
            'content'       => $validated['content'],
            'is_voice_note' => !empty($validated['is_voice_note']),
        ]);

        return response()->json([
            'success' => true,
            'comment' => [
                'id'            => $comment->id,
                'content'       => $comment->content,
                'is_voice_note' => $comment->is_voice_note,
                'user'          => ['name' => Auth::user()->name],
                'created_at'    => $comment->created_at->diffForHumans(),
            ],
        ]);
    }

    // ─── Private ─────────────────────────────────────────────────────────────

    private function validateTask(Request $request, ?int $taskId = null): array
    {
        return $request->validate([
            'title'                => 'required|string|max:255',
            'description'          => 'nullable|string',
            'due_date'             => 'nullable|date',
            'reminder_at'          => 'nullable|date',
            'task_status_id'       => 'nullable|exists:task_statuses,id',
            'task_priority_id'     => 'nullable|exists:task_priorities,id',
            'assigned_to_user_id'  => 'nullable|exists:users,id',
            'parent_task_id'       => ['nullable', 'exists:tasks,id', function ($attr, $value, $fail) use ($taskId) {
                if ($taskId && $value == $taskId) $fail('A task cannot be its own sub-task.');
            }],
            'taskable_type'        => 'nullable|string',
            'taskable_id'          => 'nullable|integer',
            'estimated_hours'      => 'nullable|numeric|min:0',
            'actual_hours'         => 'nullable|numeric|min:0',
            'is_milestone'         => 'nullable|boolean',
            'task_order'           => 'nullable|integer|min:0',
        ]);
    }

    private function authorizeTask(Task $task): void
    {
        abort_unless($task->user_id === Auth::id(), 403);
    }

    private function basicNlpParse(string $input, $priorities): array
    {
        $parsed = ['title' => $input];

        // Priority keywords
        $priorityMap = ['urgent' => 4, 'high' => 3, 'medium' => 2, 'low' => 1];
        foreach ($priorityMap as $keyword => $level) {
            if (str_contains(strtolower($input), $keyword)) {
                $match = $priorities->firstWhere('level', $level);
                if ($match) {
                    $parsed['priority_id'] = $match->id;
                    break;
                }
            }
        }

        // Tomorrow / today
        if (str_contains(strtolower($input), 'tomorrow')) {
            $parsed['due_date'] = now()->addDay()->startOfDay()->toIso8601String();
        } elseif (preg_match('/\btoday\b/i', $input)) {
            $parsed['due_date'] = now()->endOfDay()->toIso8601String();
        }

        // Time extraction: "at 3pm", "at 14:00"
        if (preg_match('/at (\d{1,2}(?::\d{2})?\s*(?:am|pm)?)/i', $input, $matches)) {
            try {
                $time = \Carbon\Carbon::parse($matches[1]);
                if (isset($parsed['due_date'])) {
                    $base = \Carbon\Carbon::parse($parsed['due_date']);
                    $parsed['due_date'] = $base->setTime($time->hour, $time->minute)->toIso8601String();
                }
            } catch (\Throwable) {}
        }

        return $parsed;
    }

    private function seedDefaults(int $userId): void
    {
        $statuses = [
            ['name' => 'To Do',       'color' => '#6366f1', 'position' => 0, 'is_default' => true,  'is_completed_status' => false],
            ['name' => 'In Progress', 'color' => '#f59e0b', 'position' => 1, 'is_default' => false, 'is_completed_status' => false],
            ['name' => 'In Review',   'color' => '#8b5cf6', 'position' => 2, 'is_default' => false, 'is_completed_status' => false],
            ['name' => 'Done',        'color' => '#22c55e', 'position' => 3, 'is_default' => false, 'is_completed_status' => true],
        ];
        foreach ($statuses as $s) {
            TaskStatus::create(array_merge($s, ['user_id' => $userId]));
        }

        $priorities = [
            ['name' => 'Low',    'color' => '#22c55e', 'level' => 1, 'is_default' => false],
            ['name' => 'Medium', 'color' => '#f59e0b', 'level' => 2, 'is_default' => true],
            ['name' => 'High',   'color' => '#f97316', 'level' => 3, 'is_default' => false],
            ['name' => 'Urgent', 'color' => '#ef4444', 'level' => 4, 'is_default' => false],
        ];
        foreach ($priorities as $p) {
            TaskPriority::create(array_merge($p, ['user_id' => $userId]));
        }
    }
}
