<?php

namespace Modules\PMCore\app\Models;

use App\Models\User;
use App\Traits\UserActionsTrait;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\SoftDeletes;
use Modules\PMCore\app\Enums\TimesheetStatus;
use OwenIt\Auditing\Auditable as AuditableTrait;
use OwenIt\Auditing\Contracts\Auditable;

class Timesheet extends Model implements Auditable
{
    use AuditableTrait, HasFactory, SoftDeletes, UserActionsTrait;

    protected $table = 'timesheets';

    protected $fillable = [
        'user_id',
        'project_id',
        'task_id',
        'date',
        'hours',
        'description',
        'is_billable',
        'billing_rate',
        'cost_rate',
        'cost_amount',
        'billable_amount',
        'status',
        'approved_by_id',
        'approved_at',
    ];

    protected $casts = [
        'date' => 'date',
        'hours' => 'decimal:2',
        'billing_rate' => 'decimal:2',
        'cost_rate' => 'decimal:2',
        'cost_amount' => 'decimal:2',
        'billable_amount' => 'decimal:2',
        'is_billable' => 'boolean',
        'approved_at' => 'datetime',
        'status' => TimesheetStatus::class,
    ];

    /**
     * Boot the model
     */
    protected static function boot()
    {
        parent::boot();

        // Auto-calculate amounts on creating and updating
        static::creating(function ($timesheet) {
            $timesheet->calculateAmounts();
        });

        static::updating(function ($timesheet) {
            $timesheet->calculateAmounts();
        });

        // Update project financials after timesheet is saved
        static::created(function ($timesheet) {
            if ($timesheet->project_id) {
                $timesheet->updateProjectFinancials();
            }
        });

        static::updated(function ($timesheet) {
            if ($timesheet->project_id) {
                $timesheet->updateProjectFinancials();
            }
        });

        static::deleted(function ($timesheet) {
            if ($timesheet->project_id) {
                $timesheet->updateProjectFinancials();
            }
        });
    }

    /**
     * Calculate cost and billable amounts
     */
    public function calculateAmounts(): void
    {
        // Calculate cost amount
        $this->cost_amount = $this->hours * ($this->cost_rate ?: 0);

        // Calculate billable amount
        if ($this->is_billable) {
            $this->billable_amount = $this->hours * ($this->billing_rate ?: 0);
        } else {
            $this->billable_amount = 0;
        }
    }

    /**
     * Update project financial values based on timesheets
     */
    protected function updateProjectFinancials(): void
    {
        try {
            if (! $this->project_id) {
                return;
            }

            $project = $this->project()->first();
            if (! $project) {
                return;
            }

            // Calculate total cost from approved/submitted timesheets
            $totalCost = $project->timesheets()
                ->whereIn('status', ['approved', 'submitted'])
                ->sum(\Illuminate\Support\Facades\DB::raw('hours * COALESCE(cost_rate, 0)'));

            // Calculate total revenue from approved/submitted billable timesheets
            $totalRevenue = $project->timesheets()
                ->whereIn('status', ['approved', 'submitted'])
                ->where('is_billable', true)
                ->sum(\Illuminate\Support\Facades\DB::raw('hours * COALESCE(billing_rate, 0)'));

            // Update project without triggering events to avoid infinite loops
            $project->timestamps = false;
            $project->update([
                'actual_cost' => $totalCost,
                'actual_revenue' => $totalRevenue,
            ]);
            $project->timestamps = true;

        } catch (\Exception $e) {
            \Illuminate\Support\Facades\Log::error('Failed to update project financials: '.$e->getMessage(), [
                'timesheet_id' => $this->id,
                'project_id' => $this->project_id,
            ]);
        }
    }

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function project(): BelongsTo
    {
        return $this->belongsTo(Project::class);
    }

    public function task(): BelongsTo
    {
        // Return a relationship to CRM Task if available, otherwise return empty relationship
        if (class_exists('\Modules\CRMCore\app\Models\Task')) {
            return $this->belongsTo(\Modules\CRMCore\app\Models\Task::class);
        }

        // Return empty relationship if CRMCore is not available
        return $this->belongsTo(User::class, 'task_id', 'non_existent_column')
            ->whereRaw('1 = 0');
    }

    public function approvedBy(): BelongsTo
    {
        return $this->belongsTo(User::class, 'approved_by_id');
    }

    public function createdBy(): BelongsTo
    {
        return $this->belongsTo(User::class, 'created_by_id');
    }

    public function updatedBy(): BelongsTo
    {
        return $this->belongsTo(User::class, 'updated_by_id');
    }

    // Scopes
    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeForProject($query, $projectId)
    {
        return $query->where('project_id', $projectId);
    }

    public function scopeForDate($query, $date)
    {
        return $query->whereDate('date', $date);
    }

    public function scopeForDateRange($query, $startDate, $endDate)
    {
        return $query->whereBetween('date', [$startDate, $endDate]);
    }

    public function scopeBillable($query)
    {
        return $query->where('is_billable', true);
    }

    public function scopeByStatus($query, $status)
    {
        return $query->where('status', $status);
    }

    public function scopeApproved($query)
    {
        return $query->where('status', TimesheetStatus::APPROVED);
    }

    public function scopePending($query)
    {
        return $query->where('status', TimesheetStatus::SUBMITTED);
    }

    // Accessors
    public function getFormattedHoursAttribute()
    {
        return number_format($this->hours, 2).' hrs';
    }

    public function getTotalAmountAttribute()
    {
        return $this->hours * ($this->billing_rate ?? 0);
    }

    public function getTotalCostAttribute()
    {
        return $this->hours * ($this->cost_rate ?? 0);
    }

    public function getStatusBadgeAttribute()
    {
        $statusColors = [
            'draft' => 'secondary',
            'submitted' => 'warning',
            'approved' => 'success',
            'rejected' => 'danger',
            'invoiced' => 'info',
        ];

        $color = $statusColors[$this->status->value] ?? 'secondary';

        return '<span class="badge bg-'.$color.'">'.ucfirst($this->status->value).'</span>';
    }

    // Methods
    public function canBeEditedBy(User $user): bool
    {
        // Users can edit their own draft timesheets
        if ($this->user_id === $user->id && $this->status === TimesheetStatus::DRAFT) {
            return true;
        }

        // Project managers can edit timesheets for their projects
        if ($this->project->project_manager_id === $user->id) {
            return true;
        }

        // Admins can edit any timesheet
        return $user->hasRole('admin') || $user->hasRole('super_admin');
    }

    public function canBeApprovedBy(User $user): bool
    {
        // Can't approve own timesheets
        if ($this->user_id === $user->id) {
            \Illuminate\Support\Facades\Log::debug('Timesheet approval denied: User trying to approve own timesheet', [
                'timesheet_id' => $this->id,
                'user_id' => $user->id,
            ]);

            return false;
        }

        // Only submitted timesheets can be approved
        if ($this->status !== TimesheetStatus::SUBMITTED) {
            \Illuminate\Support\Facades\Log::debug('Timesheet approval denied: Wrong status', [
                'timesheet_id' => $this->id,
                'status' => $this->status->value,
                'expected' => TimesheetStatus::SUBMITTED->value,
            ]);

            return false;
        }

        // Project managers can approve timesheets for their projects
        if ($this->project && $this->project->project_manager_id === $user->id) {
            return true;
        }

        // Admins can approve any timesheet
        // Check multiple variations of admin role names
        $adminRoles = ['admin', 'super_admin', 'super admin', 'Admin', 'Super Admin'];
        foreach ($adminRoles as $role) {
            if ($user->hasRole($role)) {
                return true;
            }
        }

        // Also check if user has any role containing 'admin'
        foreach ($user->getRoleNames() as $roleName) {
            if (stripos($roleName, 'admin') !== false) {
                return true;
            }
        }

        \Illuminate\Support\Facades\Log::debug('Timesheet approval denied: User lacks admin role', [
            'timesheet_id' => $this->id,
            'user_id' => $user->id,
            'user_roles' => $user->getRoleNames()->toArray(),
        ]);

        return false;
    }

    public function approve(User $approver): bool
    {
        if (! $this->canBeApprovedBy($approver)) {
            return false;
        }

        $this->update([
            'status' => TimesheetStatus::APPROVED,
            'approved_by_id' => $approver->id,
            'approved_at' => now(),
        ]);

        return true;
    }

    public function reject(User $approver): bool
    {
        if (! $this->canBeApprovedBy($approver)) {
            return false;
        }

        $this->update([
            'status' => TimesheetStatus::REJECTED,
            'approved_by_id' => $approver->id,
            'approved_at' => now(),
        ]);

        return true;
    }

    public function submit(): bool
    {
        if ($this->status !== TimesheetStatus::DRAFT) {
            return false;
        }

        $this->update(['status' => TimesheetStatus::SUBMITTED]);

        return true;
    }
}
