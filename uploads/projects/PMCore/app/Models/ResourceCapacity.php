<?php

namespace Modules\PMCore\app\Models;

use App\Models\User;
use Carbon\Carbon;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ResourceCapacity extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'date',
        'available_hours',
        'allocated_hours',
        'utilized_hours',
        'is_working_day',
        'leave_type',
        'notes',
    ];

    protected $casts = [
        'date' => 'date',
        'available_hours' => 'decimal:1',
        'allocated_hours' => 'decimal:1',
        'utilized_hours' => 'decimal:1',
        'is_working_day' => 'boolean',
    ];

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    // Scopes
    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    public function scopeForDate($query, $date)
    {
        return $query->whereDate('date', $date);
    }

    public function scopeForDateRange($query, $startDate, $endDate)
    {
        return $query->whereBetween('date', [$startDate, $endDate]);
    }

    public function scopeWorkingDays($query)
    {
        return $query->where('is_working_day', true);
    }

    public function scopeAvailable($query)
    {
        return $query->where('is_working_day', true)
            ->whereColumn('allocated_hours', '<', 'available_hours');
    }

    // Accessors
    public function getRemainingHoursAttribute()
    {
        return max(0, $this->available_hours - $this->allocated_hours);
    }

    public function getUtilizationPercentageAttribute()
    {
        if ($this->available_hours == 0) {
            return 0;
        }

        return round(($this->utilized_hours / $this->available_hours) * 100, 2);
    }

    public function getAllocationPercentageAttribute()
    {
        if ($this->available_hours == 0) {
            return 0;
        }

        return round(($this->allocated_hours / $this->available_hours) * 100, 2);
    }

    public function getIsOverallocatedAttribute()
    {
        return $this->allocated_hours > $this->available_hours;
    }

    public function getIsFullyAllocatedAttribute()
    {
        return $this->allocated_hours >= $this->available_hours;
    }

    // Methods
    public static function generateForUser($userId, Carbon $startDate, Carbon $endDate): void
    {
        $user = User::findOrFail($userId);
        $currentDate = $startDate->copy();

        while ($currentDate <= $endDate) {
            // Check if capacity already exists
            $capacity = self::firstOrNew([
                'user_id' => $userId,
                'date' => $currentDate->format('Y-m-d'),
            ]);

            if (! $capacity->exists) {
                $capacity->available_hours = $currentDate->isWeekday() ? 8.0 : 0;
                $capacity->is_working_day = $currentDate->isWeekday();
                $capacity->allocated_hours = 0;
                $capacity->utilized_hours = 0;

                // Check for holidays or leaves here if integrated with HR module
                // This is a placeholder for holiday/leave checking

                $capacity->save();
            }

            $currentDate->addDay();
        }
    }

    public static function updateAllocatedHours($userId, Carbon $date): void
    {
        $capacity = self::firstOrCreate([
            'user_id' => $userId,
            'date' => $date->format('Y-m-d'),
        ], [
            'available_hours' => $date->isWeekday() ? 8.0 : 0,
            'is_working_day' => $date->isWeekday(),
            'allocated_hours' => 0,
            'utilized_hours' => 0,
        ]);

        // Calculate total allocated hours from active allocations
        $allocations = ResourceAllocation::where('user_id', $userId)
            ->whereIn('status', ['active', 'planned'])
            ->where('start_date', '<=', $date)
            ->where(function ($q) use ($date) {
                $q->where('end_date', '>=', $date)
                    ->orWhereNull('end_date');
            })
            ->get();

        $totalAllocatedHours = 0;
        foreach ($allocations as $allocation) {
            $totalAllocatedHours += $allocation->daily_allocated_hours;
        }

        $capacity->allocated_hours = $totalAllocatedHours;
        $capacity->save();
    }

    public static function updateUtilizedHours($userId, Carbon $date): void
    {
        $capacity = self::forUser($userId)->forDate($date)->first();

        if (! $capacity) {
            return;
        }

        // Calculate utilized hours from timesheets
        $utilizedHours = Timesheet::where('user_id', $userId)
            ->whereDate('date', $date)
            ->whereIn('status', ['approved', 'submitted'])
            ->sum('hours');

        $capacity->utilized_hours = $utilizedHours;
        $capacity->save();
    }

    public function markAsLeave($leaveType = 'leave'): void
    {
        $this->update([
            'is_working_day' => false,
            'available_hours' => 0,
            'leave_type' => $leaveType,
        ]);
    }

    public function markAsWorkingDay($hours = 8.0): void
    {
        $this->update([
            'is_working_day' => true,
            'available_hours' => $hours,
            'leave_type' => null,
        ]);
    }
}
