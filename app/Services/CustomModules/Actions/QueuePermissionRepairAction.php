<?php

namespace App\Services\CustomModules\Actions;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Schema;

class QueuePermissionRepairAction
{
    /**
     * Describe this repair action
     */
    public function describe(): array
    {
        return [
            'name' => 'repair_permissions',
            'title' => 'Repair Permissions',
            'description' => 'Sync permissions from module manifest to database',
            'estimated_duration' => '2-5 seconds',
            'reversible' => true,
            'critical' => true,
        ];
    }
    
    /**
     * Check if this action can be executed for the module
     */
    public function canExecute(string $moduleName): bool
    {
        // Check if module directory exists
        return File::exists(base_path("Modules/{$moduleName}"));
    }
    
    /**
     * Execute permission repair
     */
    public function execute(string $moduleName): array
    {
        try {
            $manifest = $this->loadManifest($moduleName);
            $permissions = $manifest['permissions'] ?? [];
            
            if (empty($permissions)) {
                return [
                    'success' => true,
                    'message' => 'No permissions to repair',
                    'inserted' => 0,
                    'updated' => 0,
                ];
            }
            
            $inserted = 0;
            $updated = 0;
            $errors = [];
            
            $keyColumn = Schema::hasColumn('permissions', 'permission_key') ? 'permission_key' : 'name';

            foreach ($permissions as $perm) {
                try {
                    $key = $perm['key'] ?? null;
                    if (!$key) {
                        $errors[] = "Permission missing key field";
                        continue;
                    }
                    
                    // Check if exists
                    $existing = DB::table('permissions')
                        ->where($keyColumn, $key)
                        ->first();
                    
                    if ($existing) {
                        // Update
                        DB::table('permissions')
                            ->where($keyColumn, $key)
                            ->update([
                                'module' => $moduleName,
                                'display_name' => $perm['label'] ?? '',
                                'updated_at' => now(),
                            ]);
                        $updated++;
                    } else {
                        // Insert
                        DB::table('permissions')->insert([
                            $keyColumn => $key,
                            'module' => $moduleName,
                            'display_name' => $perm['label'] ?? '',
                            'created_at' => now(),
                            'updated_at' => now(),
                        ]);
                        $inserted++;
                    }
                    
                } catch (\Exception $e) {
                    $errors[] = "Permission '{$key}': {$e->getMessage()}";
                }
            }
            
            return [
                'success' => empty($errors),
                'message' => "Repaired {$inserted} new and {$updated} updated permissions",
                'inserted' => $inserted,
                'updated' => $updated,
                'errors' => $errors,
                'permissions_total' => count($permissions),
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'message' => "Permission repair failed: {$e->getMessage()}",
            ];
        }
    }
    
    /**
     * Rollback permission changes
     */
    public function rollback(string $moduleName): array
    {
        try {
            $deleted = DB::table('permissions')
                ->where('module', $moduleName)
                ->delete();
            
            return [
                'success' => true,
                'message' => "Removed {$deleted} permissions",
                'deleted' => $deleted,
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'message' => "Permission rollback failed: {$e->getMessage()}",
            ];
        }
    }
    
    /**
     * Load module manifest
     */
    private function loadManifest(string $moduleName): array
    {
        $path = base_path("Modules/{$moduleName}/module.json");
        
        if (!File::exists($path)) {
            return [];
        }
        
        return json_decode(File::get($path), true) ?? [];
    }
}
