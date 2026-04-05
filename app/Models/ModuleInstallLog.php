<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleInstallLog extends BaseModel
{
    protected $table = 'module_install_logs';

    protected $guarded = ['id'];

    protected $casts = [
        'blocking_issues' => 'array',
        'warnings' => 'array',
        'checks' => 'array',
        'package_summary' => 'array',
        // Enhanced casting for new fields
        'manifest_data' => 'array',
        'validation_results' => 'array',
        'pre_install_snapshot' => 'array',
        'applied_repairs' => 'array',
        'repair_results' => 'array',
        'installed_at' => 'datetime',
        'last_verified_at' => 'datetime',
        'can_rollback' => 'boolean',
    ];
    
    /**
     * Get all installations for a module
     */
    public static function forModule(string $moduleName)
    {
        return static::where('module_name', $moduleName);
    }
    
    /**
     * Get latest installation for a module
     */
    public static function latestForModule(string $moduleName)
    {
        return static::forModule($moduleName)
            ->orderBy('installed_at', 'desc')
            ->first();
    }
    
    /**
     * Get all failed installations that can be rolled back
     */
    public function scopeRollbackable($query)
    {
        return $query->where('can_rollback', true)
            ->whereIn('status', ['partial_failed', 'failed', 'partial'])
            ->whereNotNull('pre_install_snapshot');
    }
    
    /**
     * Get all installations pending rollback
     */
    public function scopePendingRollback($query)
    {
        return $query->where('status', 'rollback_needed')
            ->where('can_rollback', true);
    }
    
    /**
     * Mark this installation as failed and rollback-ready
     */
    public function markFailedButRollbackable(string $reason): void
    {
        $this->update([
            'status' => 'rollback_needed',
            'can_rollback' => !empty($this->pre_install_snapshot),
            'installation_notes' => ($this->installation_notes ?? '') . " | Failed: {$reason}",
        ]);
    }
    
    /**
     * Get human-readable status
     */
    public function getStatusLabelAttribute(): string
    {
        return match($this->status) {
            'pending' => 'Pending',
            'installed' => 'Successfully Installed',
            'partial' => 'Partially Installed',
            'partial_failed' => 'Partially Failed',
            'failed' => 'Installation Failed',
            'rollback_needed' => 'Rollback Available',
            'rolled_back' => 'Rolled Back',
            'rollback_failed' => 'Rollback Failed',
            default => 'Unknown',
        };
    }
    
    /**
     * Get CSS color for status
     */
    public function getStatusColorAttribute(): string
    {
        return match($this->status) {
            'installed' => 'success',
            'pending' => 'info',
            'partial' => 'warning',
            'partial_failed', 'failed', 'rollback_needed' => 'danger',
            'rolled_back' => 'secondary',
            'rollback_failed' => 'danger',
            default => 'default',
        };
    }
}
