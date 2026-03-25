<?php

namespace Modules\PMCore\app\Models;

use App\Traits\UserActionsTrait;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;
use Illuminate\Support\Facades\Schema;
use Modules\PMCore\app\Enums\ProjectPriority;
use Modules\PMCore\app\Enums\ProjectStatus;
use Modules\PMCore\app\Enums\ProjectType;
use OwenIt\Auditing\Auditable;
use OwenIt\Auditing\Contracts\Auditable as AuditableContract;

class Project extends Model implements AuditableContract
{
    use Auditable, HasFactory, SoftDeletes, UserActionsTrait;

    protected $fillable = [
        'name',
        'code',
        'description',
        'status',
        'type',
        'start_date',
        'end_date',
        'budget',
        'actual_cost',
        'actual_revenue',
        'completion_percentage',
        'completed_at',
        'is_archived',
        'priority',
        'color_code',
        'is_billable',
        'hourly_rate',
        'client_id',
        'project_manager_id',
    ];

    protected $casts = [
        'start_date' => 'date',
        'end_date' => 'date',
        'completed_at' => 'datetime',
        'budget' => 'decimal:2',
        'actual_cost' => 'decimal:2',
        'actual_revenue' => 'decimal:2',
        'hourly_rate' => 'decimal:2',
        'is_billable' => 'boolean',
        'is_archived' => 'boolean',
        'completion_percentage' => 'integer',
        'status' => ProjectStatus::class,
        'type' => ProjectType::class,
        'priority' => ProjectPriority::class,
    ];

    protected $dates = [
        'start_date',
        'end_date',
        'created_at',
        'updated_at',
        'deleted_at',
    ];

    /**
     * The attributes that should be hidden for serialization.
     */
    protected $hidden = [];

    /**
     * Boot the model.
     */
    protected static function boot()
    {
        parent::boot();

        static::creating(function ($project) {
            if (empty($project->code)) {
                $project->code = static::generateProjectCode($project->name);
            }
        });
    }

    /**
     * Generate a unique project code based on settings.
     */
    public static function generateProjectCode(string $name): string
    {
        $settingsService = app(\App\Services\Settings\ModuleSettingsService::class);

        // Get settings
        $autoGenerate = $settingsService->get('PMCore', 'auto_generate_codes', true);
        $prefixLength = (int) $settingsService->get('PMCore', 'code_prefix_length', '3');
        $separator = $settingsService->get('PMCore', 'code_separator', '-');

        // If auto-generation is disabled, return empty string
        if (! $autoGenerate) {
            return '';
        }

        // Extract prefix from project name
        $base = strtoupper(substr(preg_replace('/[^A-Za-z0-9]/', '', $name), 0, $prefixLength));
        if (strlen($base) < $prefixLength) {
            $base = str_pad($base, $prefixLength, 'X');
        }

        $counter = 1;
        $code = $base.$separator.str_pad($counter, 3, '0', STR_PAD_LEFT);

        while (static::where('code', $code)->exists()) {
            $counter++;
            $code = $base.$separator.str_pad($counter, 3, '0', STR_PAD_LEFT);
        }

        return $code;
    }

    /**
     * Get the client (company) that owns the project.
     */
    public function client()
    {
        // Soft relationship to CRMCore Company model
        if (class_exists('\Modules\CRMCore\app\Models\Company')) {
            return $this->belongsTo(\Modules\CRMCore\app\Models\Company::class, 'client_id');
        }

        // Return empty relationship if CRMCore is not available
        return $this->belongsTo(\App\Models\User::class, 'client_id', 'non_existent_column')
            ->whereRaw('1 = 0');
    }

    /**
     * Get the project manager.
     */
    public function projectManager()
    {
        return $this->belongsTo(\App\Models\User::class, 'project_manager_id');
    }

    /**
     * Get the project members.
     */
    public function members()
    {
        return $this->hasMany(ProjectMember::class);
    }

    /**
     * Get the users assigned to this project.
     */
    public function users()
    {
        return $this->belongsToMany(\App\Models\User::class, 'project_members')
            ->withPivot(['role', 'hourly_rate', 'allocation_percentage', 'joined_at', 'left_at'])
            ->withTimestamps();
    }

    /**
     * Get active project members.
     */
    public function activeMembers()
    {
        return $this->members()->whereNull('left_at');
    }

    /**
     * Get resource allocations for this project.
     */
    public function resourceAllocations()
    {
        return $this->hasMany(ResourceAllocation::class);
    }

    /**
     * Get project tasks (CRMCore tasks using polymorphic relationship).
     */
    public function tasks()
    {
        // Use CRMCore tasks with polymorphic relationship
        if (class_exists('\Modules\CRMCore\app\Models\Task')) {
            return $this->morphMany(\Modules\CRMCore\app\Models\Task::class, 'taskable');
        }

        // Return empty relationship if not available
        return $this->hasMany(\App\Models\User::class, 'id', 'non_existent_column')->whereRaw('1 = 0');
    }

    /**
     * Get root tasks (tasks without parent) for this project.
     */
    public function rootTasks()
    {
        return $this->tasks()->rootTasks();
    }

    /**
     * Get milestone tasks for this project.
     */
    public function milestones()
    {
        return $this->tasks()->milestones();
    }

    /**
     * Get completed tasks for this project.
     */
    public function completedTasks()
    {
        return $this->tasks()->completed();
    }

    /**
     * Get pending tasks for this project.
     */
    public function pendingTasks()
    {
        return $this->tasks()->pending();
    }

    /**
     * Get overdue tasks for this project.
     */
    public function overdueTasks()
    {
        return $this->tasks()->where('due_date', '<', now())->whereHas('status', function ($query) {
            $query->whereNotIn('name', ['Completed', 'Done']);
        });
    }

    /**
     * Get project completion percentage based on tasks.
     */
    public function getCompletionPercentage()
    {
        $totalTasks = $this->tasks()->count();
        if ($totalTasks === 0) {
            return 0;
        }

        $completedTasks = $this->completedTasks()->count();

        return round(($completedTasks / $totalTasks) * 100, 2);
    }

    /**
     * Get total estimated hours for project tasks.
     */
    public function getTotalEstimatedHours()
    {
        return $this->tasks()->sum('estimated_hours') ?? 0;
    }

    /**
     * Get total actual hours for project tasks.
     */
    public function getTotalActualHours()
    {
        return $this->tasks()->sum('actual_hours') ?? 0;
    }

    /**
     * Get project timesheets.
     */
    public function timesheets()
    {
        return $this->hasMany(Timesheet::class);
    }

    /**
     * Get project events (if Calendar is available).
     */
    public function events()
    {
        if (class_exists('\Modules\Calendar\app\Models\Event')) {
            return $this->morphMany(\Modules\Calendar\app\Models\Event::class, 'related');
        }

        return null;
    }

    /**
     * Scope for filtering by status.
     */
    public function scopeByStatus($query, $status)
    {
        return $query->where('status', $status);
    }

    /**
     * Scope for filtering by type.
     */
    public function scopeByType($query, $type)
    {
        return $query->where('type', $type);
    }

    /**
     * Scope for active projects.
     */
    public function scopeActive($query)
    {
        return $query->whereIn('status', [
            ProjectStatus::PLANNING->value,
            ProjectStatus::IN_PROGRESS->value,
        ]);
    }

    /**
     * Scope for completed projects.
     */
    public function scopeCompleted($query)
    {
        return $query->where('status', ProjectStatus::COMPLETED->value);
    }

    /**
     * Scope for projects managed by user.
     */
    public function scopeManagedBy($query, $userId)
    {
        return $query->where('project_manager_id', $userId);
    }

    /**
     * Scope for projects with user as member.
     */
    public function scopeWithMember($query, $userId)
    {
        return $query->whereHas('members', function ($q) use ($userId) {
            $q->where('user_id', $userId)->whereNull('left_at');
        });
    }

    /**
     * Get client name for display.
     */
    public function getClientNameAttribute(): ?string
    {
        return $this->client?->name;
    }

    /**
     * Get project manager name for display.
     */
    public function getProjectManagerNameAttribute(): ?string
    {
        return $this->projectManager?->getFullName();
    }

    /**
     * Get status label for display.
     */
    public function getStatusLabelAttribute(): string
    {
        return $this->status->label();
    }

    /**
     * Get type label for display.
     */
    public function getTypeLabelAttribute(): string
    {
        return $this->type->label();
    }

    /**
     * Get priority label for display.
     */
    public function getPriorityLabelAttribute(): string
    {
        return $this->priority->label();
    }

    /**
     * Get project progress percentage.
     */
    public function getProgressPercentage(): float
    {
        // Check if TaskSystem integration is available and properly configured
        try {
            if (
                class_exists('\\Modules\\TaskSystem\\app\\Models\\Task') &&
                Schema::hasTable('tasks') &&
                Schema::hasColumn('tasks', 'project_id')
            ) {

                $totalTasks = $this->tasks()->count();
                if ($totalTasks === 0) {
                    return 0;
                }
                $completedTasks = $this->tasks()->where('status', 'completed')->count();

                return round(($completedTasks / $totalTasks) * 100, 2);
            }
        } catch (\Exception $e) {
            // If there's any error with task integration, return 0
            \Log::debug('Project progress calculation failed: '.$e->getMessage());
        }

        // Return default progress based on project status if no task integration
        switch ($this->status) {
            case ProjectStatus::PLANNING:
                return 0;
            case ProjectStatus::IN_PROGRESS:
                return 50;
            case ProjectStatus::COMPLETED:
                return 100;
            case ProjectStatus::ON_HOLD:
                return 25;
            case ProjectStatus::CANCELLED:
                return 0;
            default:
                return 0;
        }
    }

    /**
     * Check if project is overdue.
     */
    public function isOverdue(): bool
    {
        if (! $this->end_date) {
            return false;
        }

        return $this->end_date->isPast() &&
          ! in_array($this->status->value, [ProjectStatus::COMPLETED->value, ProjectStatus::CANCELLED->value]);
    }

    /**
     * Get days until deadline.
     */
    public function getDaysUntilDeadline(): ?int
    {
        if (! $this->end_date) {
            return null;
        }

        return now()->diffInDays($this->end_date, false);
    }

    /**
     * Get project duration in days.
     */
    public function getDurationInDays(): ?int
    {
        if (! $this->start_date || ! $this->end_date) {
            return null;
        }

        return $this->start_date->diffInDays($this->end_date);
    }

    /**
     * Check if user is project manager.
     */
    public function isManageredBy($userId): bool
    {
        return $this->project_manager_id == $userId;
    }

    /**
     * Check if user is project member.
     */
    public function hasMember($userId): bool
    {
        return $this->members()
            ->where('user_id', $userId)
            ->whereNull('left_at')
            ->exists();
    }

    /**
     * Get user's role in project.
     */
    public function getUserRole($userId): ?string
    {
        if ($this->isManageredBy($userId)) {
            return 'manager';
        }

        $member = $this->members()
            ->where('user_id', $userId)
            ->whereNull('left_at')
            ->first();

        return $member?->role;
    }

    /**
     * Add member to project.
     */
    public function addMember($userId, array $attributes = []): ProjectMember
    {
        return $this->members()->create(array_merge([
            'user_id' => $userId,
            'joined_at' => now(),
        ], $attributes));
    }

    /**
     * Remove member from project.
     */
    public function removeMember($userId): bool
    {
        return $this->members()
            ->where('user_id', $userId)
            ->whereNull('left_at')
            ->update(['left_at' => now()]);
    }

    /**
     * Calculate and update actual cost from timesheets.
     */
    public function updateActualCost(): void
    {
        $totalCost = 0;

        // Calculate cost from timesheets
        if ($this->timesheets()->exists()) {
            $totalCost = $this->timesheets()
                ->where('status', 'approved')
                ->sum('cost_amount');
        }

        $this->update(['actual_cost' => $totalCost]);
    }

    /**
     * Calculate and update actual revenue from invoices.
     */
    public function updateActualRevenue(): void
    {
        $totalRevenue = 0;

        // Calculate revenue from timesheets (billable hours)
        if ($this->timesheets()->exists()) {
            $totalRevenue = $this->timesheets()
                ->where('status', 'approved')
                ->where('is_billable', true)
                ->sum('billable_amount');
        }

        $this->update(['actual_revenue' => $totalRevenue]);
    }

    /**
     * Get budget variance (budget - actual cost).
     */
    public function getBudgetVarianceAttribute(): float
    {
        return $this->budget - $this->actual_cost;
    }

    /**
     * Get budget variance percentage.
     */
    public function getBudgetVariancePercentageAttribute(): float
    {
        if ($this->budget == 0) {
            return 0;
        }

        return round((($this->budget - $this->actual_cost) / $this->budget) * 100, 2);
    }

    /**
     * Get profit margin (revenue - cost).
     */
    public function getProfitMarginAttribute(): float
    {
        return $this->actual_revenue - $this->actual_cost;
    }

    /**
     * Get profit margin percentage.
     */
    public function getProfitMarginPercentageAttribute(): float
    {
        if ($this->actual_revenue == 0) {
            return 0;
        }

        return round((($this->actual_revenue - $this->actual_cost) / $this->actual_revenue) * 100, 2);
    }

    /**
     * Check if project is over budget.
     */
    public function isOverBudget(): bool
    {
        return $this->budget > 0 && $this->actual_cost > $this->budget;
    }

    /**
     * Mark project as completed.
     */
    public function markAsCompleted(): void
    {
        $this->update([
            'status' => ProjectStatus::COMPLETED,
            'completion_percentage' => 100,
            'completed_at' => now(),
        ]);
    }

    /**
     * Archive the project.
     */
    public function archive(): void
    {
        $this->update(['is_archived' => true]);
    }

    /**
     * Unarchive the project.
     */
    public function unarchive(): void
    {
        $this->update(['is_archived' => false]);
    }

    /**
     * Scope for non-archived projects.
     */
    public function scopeNotArchived($query)
    {
        return $query->where('is_archived', false);
    }

    /**
     * Scope for archived projects.
     */
    public function scopeArchived($query)
    {
        return $query->where('is_archived', true);
    }

    /**
     * Scope for ongoing projects.
     */
    public function scopeOngoing($query)
    {
        return $query->whereIn('status', [ProjectStatus::IN_PROGRESS, ProjectStatus::PLANNING]);
    }

    /**
     * Calculate total hours from timesheets.
     */
    public function getTotalHoursAttribute(): float
    {
        return $this->timesheets()
            ->where('status', 'approved')
            ->sum('hours');
    }

    /**
     * Calculate billable hours from timesheets.
     */
    public function getBillableHoursAttribute(): float
    {
        return $this->timesheets()
            ->where('status', 'approved')
            ->where('is_billable', true)
            ->sum('hours');
    }
}
