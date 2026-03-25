<?php

namespace App\Models\Tasks;

use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\MorphTo;
use Illuminate\Database\Eloquent\SoftDeletes;

class Task extends Model
{
    use SoftDeletes;

    protected $table = 'tasks';

    protected $fillable = [
        'user_id',
        'title',
        'description',
        'due_date',
        'completed_at',
        'reminder_at',
        'task_status_id',
        'task_priority_id',
        'assigned_to_user_id',
        'completed_by_user_id',
        'taskable_id',
        'taskable_type',
        'parent_task_id',
        'estimated_hours',
        'actual_hours',
        'task_order',
        'is_milestone',
        'ai_metadata',
    ];

    protected $casts = [
        'due_date'       => 'datetime',
        'completed_at'   => 'datetime',
        'reminder_at'    => 'datetime',
        'is_milestone'   => 'boolean',
        'estimated_hours'=> 'decimal:2',
        'actual_hours'   => 'decimal:2',
        'ai_metadata'    => 'array',
    ];

    // ─── Relationships ───────────────────────────────────────────────────────

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function status(): BelongsTo
    {
        return $this->belongsTo(TaskStatus::class, 'task_status_id');
    }

    public function priority(): BelongsTo
    {
        return $this->belongsTo(TaskPriority::class, 'task_priority_id');
    }

    public function assignedTo(): BelongsTo
    {
        return $this->belongsTo(User::class, 'assigned_to_user_id');
    }

    public function completedBy(): BelongsTo
    {
        return $this->belongsTo(User::class, 'completed_by_user_id');
    }

    /** Polymorphic: Contact, Company, Lead, Deal, Project, Job */
    public function taskable(): MorphTo
    {
        return $this->morphTo();
    }

    public function parentTask(): BelongsTo
    {
        return $this->belongsTo(Task::class, 'parent_task_id');
    }

    public function subTasks(): HasMany
    {
        return $this->hasMany(Task::class, 'parent_task_id');
    }

    public function comments(): HasMany
    {
        return $this->hasMany(TaskComment::class)->latest();
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    public function isCompleted(): bool
    {
        return $this->completed_at !== null;
    }

    public function markCompleted(int $byUserId): void
    {
        $this->update([
            'completed_at'       => now(),
            'completed_by_user_id' => $byUserId,
        ]);

        // If status has is_completed_status flag, keep it; otherwise auto-select first completed status
        if ($this->status && !$this->status->is_completed_status) {
            $completedStatus = TaskStatus::forUser($this->user_id)
                ->where('is_completed_status', true)
                ->first();
            if ($completedStatus) {
                $this->update(['task_status_id' => $completedStatus->id]);
            }
        }
    }

    public function markIncomplete(): void
    {
        $this->update(['completed_at' => null, 'completed_by_user_id' => null]);
    }

    public function getIsOverdueAttribute(): bool
    {
        return $this->due_date && $this->due_date->isPast() && !$this->isCompleted();
    }

    // ─── Scopes ──────────────────────────────────────────────────────────────

    public function scopeForUser($query, int $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeCompleted($query)
    {
        return $query->whereNotNull('completed_at');
    }

    public function scopePending($query)
    {
        return $query->whereNull('completed_at');
    }

    public function scopeOverdue($query)
    {
        return $query->whereNull('completed_at')
            ->whereNotNull('due_date')
            ->where('due_date', '<', now());
    }

    public function scopeRootTasks($query)
    {
        return $query->whereNull('parent_task_id');
    }

    public function scopeMilestones($query)
    {
        return $query->where('is_milestone', true);
    }

    public function scopeDueToday($query)
    {
        return $query->whereNull('completed_at')
            ->whereDate('due_date', today());
    }
}
