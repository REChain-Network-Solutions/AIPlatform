<?php

namespace Modules\PMCore\app\Models;

use App\Models\User;
use App\Traits\UserActionsTrait;
use Carbon\Carbon;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\SoftDeletes;

class ResourceAllocation extends Model
{
    use HasFactory, SoftDeletes, UserActionsTrait;

    protected $fillable = [
        'user_id',
        'project_id',
        'start_date',
        'end_date',
        'allocation_percentage',
        'hours_per_day',
        'allocation_type',
        'task_id',
        'phase',
        'notes',
        'is_billable',
        'is_confirmed',
        'status',
        'created_by_id',
        'updated_by_id',
    ];

    protected $casts = [
        'start_date' => 'date',
        'end_date' => 'date',
        'allocation_percentage' => 'decimal:2',
        'hours_per_day' => 'decimal:1',
        'is_billable' => 'boolean',
        'is_confirmed' => 'boolean',
    ];

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
        if (class_exists('\\Modules\\CRMCore\\app\\Models\\Task')) {
            return $this->belongsTo('\\Modules\\CRMCore\\app\\Models\\Task', 'task_id');
        }

        return $this->belongsTo(Project::class, 'task_id')->whereRaw('1=0'); // Empty relation
    }

    // Scopes
    public function scopeActive($query)
    {
        return $query->where('status', 'active');
    }

    public function scopePlanned($query)
    {
        return $query->where('status', 'planned');
    }

    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeForProject($query, $projectId)
    {
        return $query->where('project_id', $projectId);
    }

    public function scopeInDateRange($query, $startDate, $endDate)
    {
        return $query->where(function ($q) use ($startDate, $endDate) {
            $q->whereBetween('start_date', [$startDate, $endDate])
                ->orWhereBetween('end_date', [$startDate, $endDate])
                ->orWhere(function ($q2) use ($startDate, $endDate) {
                    $q2->where('start_date', '<=', $startDate)
                        ->where(function ($q3) use ($endDate) {
                            $q3->where('end_date', '>=', $endDate)
                                ->orWhereNull('end_date');
                        });
                });
        });
    }

    public function scopeCurrentAndFuture($query)
    {
        return $query->where(function ($q) {
            $q->where('end_date', '>=', now()->startOfDay())
                ->orWhereNull('end_date');
        });
    }

    // Accessors
    public function getDailyAllocatedHoursAttribute()
    {
        return ($this->hours_per_day * $this->allocation_percentage) / 100;
    }

    public function getWeeklyAllocatedHoursAttribute()
    {
        return $this->daily_allocated_hours * 5; // Assuming 5-day work week
    }

    public function getMonthlyAllocatedHoursAttribute()
    {
        return $this->daily_allocated_hours * 22; // Assuming 22 working days per month
    }

    public function getIsActiveAttribute()
    {
        $today = now()->startOfDay();

        return $this->start_date <= $today &&
               ($this->end_date === null || $this->end_date >= $today) &&
               $this->status === 'active';
    }

    public function getDurationInDaysAttribute()
    {
        if (! $this->end_date) {
            return null;
        }

        return $this->start_date->diffInDays($this->end_date) + 1;
    }

    public function getTotalAllocatedHoursAttribute()
    {
        if (! $this->duration_in_days) {
            return null;
        }

        // Calculate working days between dates
        $workingDays = 0;
        $currentDate = $this->start_date->copy();

        while ($currentDate <= $this->end_date) {
            if ($currentDate->isWeekday()) {
                $workingDays++;
            }
            $currentDate->addDay();
        }

        return $workingDays * $this->daily_allocated_hours;
    }

    // Methods
    public function isOverlapping(ResourceAllocation $other): bool
    {
        // Check if same user
        if ($this->user_id !== $other->user_id) {
            return false;
        }

        // Check date overlap
        $thisEnd = $this->end_date ?? Carbon::parse('2099-12-31');
        $otherEnd = $other->end_date ?? Carbon::parse('2099-12-31');

        return $this->start_date <= $otherEnd && $thisEnd >= $other->start_date;
    }

    public function getOverlapPercentage(ResourceAllocation $other): float
    {
        if (! $this->isOverlapping($other)) {
            return 0;
        }

        return $this->allocation_percentage + $other->allocation_percentage;
    }

    public function canBeEditedBy(User $user): bool
    {
        // Resource can edit their own future allocations if not confirmed
        if ($this->user_id === $user->id && ! $this->is_confirmed && $this->status === 'planned') {
            return true;
        }

        // Project manager can edit allocations
        if ($this->project->project_manager_id === $user->id) {
            return true;
        }

        // Admins can edit any allocation
        return $user->hasRole('admin') || $user->hasRole('super_admin');
    }

    public function confirm(): bool
    {
        if ($this->is_confirmed) {
            return false;
        }

        $this->update([
            'is_confirmed' => true,
            'status' => 'active',
        ]);

        return true;
    }

    public function cancel(): bool
    {
        if ($this->status === 'completed') {
            return false;
        }

        $this->update(['status' => 'cancelled']);

        return true;
    }

    public function checkCapacityConflicts(): array
    {
        $conflicts = [];

        // Get overlapping allocations for the same user
        $overlapping = self::where('user_id', $this->user_id)
            ->where('id', '!=', $this->id)
            ->whereIn('status', ['planned', 'active'])
            ->inDateRange($this->start_date, $this->end_date ?? now()->addYear())
            ->get();

        foreach ($overlapping as $allocation) {
            $totalPercentage = $this->getOverlapPercentage($allocation);
            if ($totalPercentage > 100) {
                $conflicts[] = [
                    'allocation' => $allocation,
                    'total_percentage' => $totalPercentage,
                    'overlap_start' => max($this->start_date, $allocation->start_date),
                    'overlap_end' => min(
                        $this->end_date ?? Carbon::parse('2099-12-31'),
                        $allocation->end_date ?? Carbon::parse('2099-12-31')
                    ),
                ];
            }
        }

        return $conflicts;
    }
}
