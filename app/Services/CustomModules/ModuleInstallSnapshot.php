<?php

namespace App\Services\CustomModules;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Schema;

class ModuleInstallSnapshot
{
    /**
     * Capture system state BEFORE module installation
     */
    public function snapshotBefore(string $moduleName): array
    {
        return [
            'permissions' => $this->snapshotPermissions($moduleName),
            'module_settings' => $this->snapshotModuleSettings($moduleName),
            'timestamp' => now()->toIso8601String(),
        ];
    }
    
    /**
     * Restore system state FROM a snapshot
     */
    public function restoreFrom(string $moduleName, array $snapshot): array
    {
        $results = [];
        
        try {
            // Restore permissions
            if (isset($snapshot['permissions'])) {
                $results['permissions'] = $this->restorePermissions($moduleName, $snapshot['permissions']);
            }
            
            // Restore module settings
            if (isset($snapshot['module_settings'])) {
                $results['module_settings'] = $this->restoreModuleSettings($moduleName, $snapshot['module_settings']);
            }
            
            return [
                'success' => true,
                'message' => 'State restored from snapshot',
                'results' => $results,
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'message' => "Snapshot restore failed: {$e->getMessage()}",
            ];
        }
    }
    
    /**
     * Capture current permissions for module
     */
    private function snapshotPermissions(string $moduleName): array
    {
        $keyColumn = Schema::hasColumn('permissions', 'permission_key') ? 'permission_key' : 'name';

        return DB::table('permissions')
            ->where('module', $moduleName)
            ->select($keyColumn . ' as permission_key', 'module', 'display_name')
            ->get()
            ->toArray();
    }
    
    /**
     * Capture current module settings
     */
    private function snapshotModuleSettings(string $moduleName): array
    {
        // Check if modules table exists
        if (!DB::connection()->getSchemaBuilder()->hasTable('modules')) {
            return [];
        }
        
        $module = DB::table('modules')
            ->where('module_name', $moduleName)
            ->first();
        
        if (!$module) {
            return [];
        }
        
        // Capture module settings per company
        return DB::table('module_settings')
            ->where('module_id', $module->id)
            ->select('company_id', 'setting_key', 'setting_value')
            ->get()
            ->toArray();
    }
    
    /**
     * Restore permissions from snapshot
     */
    private function restorePermissions(string $moduleName, array $snapshotPermissions): array
    {
        $keyColumn = Schema::hasColumn('permissions', 'permission_key') ? 'permission_key' : 'name';

        // Delete all current permissions for module
        $deleted = DB::table('permissions')
            ->where('module', $moduleName)
            ->delete();
        
        // Re-insert from snapshot
        $restored = 0;
        
        foreach ($snapshotPermissions as $perm) {
            $permKey = is_array($perm)
                ? ($perm['permission_key'] ?? '')
                : ($perm->permission_key ?? '');

            DB::table('permissions')->insert([
                $keyColumn => $permKey,
                'module' => $moduleName,
                'display_name' => is_array($perm) ? ($perm['display_name'] ?? '') : ($perm->display_name ?? ''),
                'created_at' => now(),
                'updated_at' => now(),
            ]);
            $restored++;
        }
        
        return [
            'deleted' => $deleted,
            'restored' => $restored,
        ];
    }
    
    /**
     * Restore module settings from snapshot
     */
    private function restoreModuleSettings(string $moduleName, array $snapshotSettings): array
    {
        if (!DB::connection()->getSchemaBuilder()->hasTable('module_settings')) {
            return ['skipped' => true, 'reason' => 'module_settings table not found'];
        }
        
        $module = DB::table('modules')
            ->where('module_name', $moduleName)
            ->first();
        
        if (!$module) {
            return ['skipped' => true, 'reason' => 'Module not registered'];
        }
        
        // Delete current settings
        $deleted = DB::table('module_settings')
            ->where('module_id', $module->id)
            ->delete();
        
        // Restore from snapshot
        $restored = 0;
        
        foreach ($snapshotSettings as $setting) {
            DB::table('module_settings')->insert([
                'module_id' => $module->id,
                'company_id' => $setting['company_id'] ?? $setting->company_id,
                'setting_key' => $setting['setting_key'] ?? $setting->setting_key,
                'setting_value' => $setting['setting_value'] ?? $setting->setting_value,
                'created_at' => now(),
                'updated_at' => now(),
            ]);
            $restored++;
        }
        
        return [
            'deleted' => $deleted,
            'restored' => $restored,
        ];
    }
}
