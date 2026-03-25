<?php

namespace Modules\PMCore\app\Models;

use App\Traits\UserActionsTrait;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\SoftDeletes;
use OwenIt\Auditing\Auditable as AuditableTrait;
use OwenIt\Auditing\Contracts\Auditable;

class ProjectStatus extends Model implements Auditable
{
    use AuditableTrait, HasFactory, SoftDeletes, UserActionsTrait;

    protected $table = 'project_statuses';

    protected $fillable = [
        'name',
        'description',
        'slug',
        'color',
        'sort_order',
        'position',
        'is_active',
        'is_default',
        'is_completed',
    ];

    protected $casts = [
        'is_active' => 'boolean',
        'is_default' => 'boolean',
        'is_completed' => 'boolean',
        'sort_order' => 'integer',
        'position' => 'integer',
    ];

    /**
     * Get the projects that have this status.
     */
    public function projects()
    {
        return $this->hasMany(Project::class, 'status', 'slug');
    }

    /**
     * Scope to get only active statuses.
     */
    public function scopeActive($query)
    {
        return $query->where('is_active', true);
    }

    /**
     * Scope to get the default status.
     */
    public function scopeDefault($query)
    {
        return $query->where('is_default', true);
    }

    /**
     * Scope to get completed statuses.
     */
    public function scopeCompleted($query)
    {
        return $query->where('is_completed', true);
    }

    /**
     * Scope to order by sort order.
     */
    public function scopeOrdered($query)
    {
        return $query->orderBy('sort_order');
    }

    /**
     * Get the color for badge display.
     */
    public function getColorAttribute($value)
    {
        return $value ?: '#6c757d';
    }

    /**
     * Get the status label for display.
     */
    public function getDisplayNameAttribute()
    {
        return $this->name;
    }

    /**
     * Check if this status represents completion.
     */
    public function isCompleted()
    {
        return $this->is_completed;
    }

    /**
     * Check if this status is the default.
     */
    public function isDefault()
    {
        return $this->is_default;
    }

    /**
     * Check if this status is active.
     */
    public function isActive()
    {
        return $this->is_active;
    }

    /**
     * Get the default status.
     */
    public static function getDefault(): ?self
    {
        return static::default()->first();
    }

    /**
     * Set as default status.
     */
    public function setAsDefault(): bool
    {
        // Remove default from all other statuses
        static::where('is_default', true)->update(['is_default' => false]);

        // Set this as default
        return $this->update(['is_default' => true]);
    }

    /**
     * Get next position for new status.
     */
    public static function getNextPosition(): int
    {
        return static::max('sort_order') + 1;
    }

    /**
     * Reorder statuses.
     */
    public static function reorder(array $orderedIds): bool
    {
        foreach ($orderedIds as $position => $id) {
            static::where('id', $id)->update(['sort_order' => $position]);
        }

        return true;
    }
}
