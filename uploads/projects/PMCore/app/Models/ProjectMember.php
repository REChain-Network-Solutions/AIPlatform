<?php

namespace Modules\PMCore\app\Models;

use App\Traits\UserActionsTrait;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Modules\PMCore\app\Enums\ProjectMemberRole;
use OwenIt\Auditing\Auditable;
use OwenIt\Auditing\Contracts\Auditable as AuditableContract;

class ProjectMember extends Model implements AuditableContract
{
    use Auditable, HasFactory, UserActionsTrait;

    protected $fillable = [
        'project_id',
        'user_id',
        'role',
        'hourly_rate',
        'allocation_percentage',
        'joined_at',
        'left_at',
    ];

    protected $casts = [
        'hourly_rate' => 'decimal:2',
        'allocation_percentage' => 'integer',
        'joined_at' => 'datetime',
        'left_at' => 'datetime',
        'role' => ProjectMemberRole::class,
    ];

    protected $dates = [
        'joined_at',
        'left_at',
        'created_at',
        'updated_at',
    ];

    /**
     * Get the project.
     */
    public function project()
    {
        return $this->belongsTo(Project::class);
    }

    /**
     * Get the user.
     */
    public function user()
    {
        return $this->belongsTo(\App\Models\User::class);
    }

    /**
     * Scope for active members.
     */
    public function scopeActive($query)
    {
        return $query->whereNull('left_at');
    }

    /**
     * Scope for members by role.
     */
    public function scopeByRole($query, $role)
    {
        return $query->where('role', $role);
    }

    /**
     * Scope for members of a specific project.
     */
    public function scopeForProject($query, $projectId)
    {
        return $query->where('project_id', $projectId);
    }

    /**
     * Scope for members assigned to a specific user.
     */
    public function scopeForUser($query, $userId)
    {
        return $query->where('user_id', $userId);
    }

    /**
     * Check if member is active.
     */
    public function isActive(): bool
    {
        return is_null($this->left_at);
    }

    /**
     * Mark member as left.
     */
    public function markAsLeft(): bool
    {
        return $this->update(['left_at' => now()]);
    }

    /**
     * Reactivate member.
     */
    public function reactivate(): bool
    {
        return $this->update(['left_at' => null]);
    }

    /**
     * Check if member has permission.
     */
    public function hasPermission(string $permission): bool
    {
        return in_array($permission, $this->role->permissions());
    }

    /**
     * Get role label for display.
     */
    public function getRoleLabelAttribute(): string
    {
        return $this->role->label();
    }

    /**
     * Get user's full name.
     */
    public function getUserNameAttribute(): ?string
    {
        return $this->user?->getFullName();
    }

    /**
     * Get project name.
     */
    public function getProjectNameAttribute(): ?string
    {
        return $this->project?->name;
    }

    /**
     * Calculate member's weekly capacity based on allocation.
     */
    public function getWeeklyCapacityHours(): float
    {
        $standardWeeklyHours = 40; // This could be configurable

        return ($this->allocation_percentage / 100) * $standardWeeklyHours;
    }

    /**
     * Get member's effective hourly rate.
     */
    public function getEffectiveHourlyRate(): ?float
    {
        // Use member-specific rate if set, otherwise use project rate, otherwise user's default rate
        if ($this->hourly_rate) {
            return $this->hourly_rate;
        }

        if ($this->project->hourly_rate) {
            return $this->project->hourly_rate;
        }

        // Could extend to get user's default hourly rate if that exists in User model
        return null;
    }
}
