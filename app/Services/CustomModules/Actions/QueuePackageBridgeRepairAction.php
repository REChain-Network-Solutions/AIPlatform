<?php

namespace App\Services\CustomModules\Actions;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;

class QueuePackageBridgeRepairAction
{
    /**
     * Describe this action
     */
    public function describe(): array
    {
        return [
            'name' => 'repair_package_bridge',
            'title' => 'Register in Package System',
            'description' => 'Register module in the package_modules table for entitlement tracking',
            'estimated_duration' => '1-2 seconds',
            'reversible' => true,
            'critical' => false,
        ];
    }
    
    /**
     * Check if action can execute
     */
    public function canExecute(string $moduleName): bool
    {
        return File::exists(base_path("Modules/{$moduleName}"));
    }
    
    /**
     * Execute package bridge repair
     */
    public function execute(string $moduleName): array
    {
        try {
            // Check if table exists
            if (!$this->tableExists('package_modules')) {
                return [
                    'success' => true,
                    'message' => 'package_modules table not found - skipping',
                    'skipped' => true,
                ];
            }
            
            // Register module for all packages that might need it
            // In a real system, this would be more granular
            $packages = DB::table('packages')->pluck('id')->toArray();
            
            $registered = 0;
            $errors = [];
            
            foreach ($packages as $packageId) {
                try {
                    // Check if already registered
                    $exists = DB::table('package_modules')
                        ->where('package_id', $packageId)
                        ->where('module_name', $moduleName)
                        ->exists();
                    
                    if (!$exists) {
                        DB::table('package_modules')->insert([
                            'package_id' => $packageId,
                            'module_name' => $moduleName,
                            'created_at' => now(),
                        ]);
                        $registered++;
                    }
                } catch (\Exception $e) {
                    $errors[] = "Package {$packageId}: {$e->getMessage()}";
                }
            }
            
            return [
                'success' => empty($errors),
                'message' => "Registered module with {$registered} packages",
                'registered' => $registered,
                'errors' => $errors,
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'message' => "Package bridge repair failed: {$e->getMessage()}",
            ];
        }
    }
    
    /**
     * Rollback package bridge changes
     */
    public function rollback(string $moduleName): array
    {
        try {
            if (!$this->tableExists('package_modules')) {
                return [
                    'success' => true,
                    'message' => 'package_modules table not found - nothing to rollback',
                ];
            }
            
            $deleted = DB::table('package_modules')
                ->where('module_name', $moduleName)
                ->delete();
            
            return [
                'success' => true,
                'message' => "Unregistered module from {$deleted} packages",
                'deleted' => $deleted,
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
            ];
        }
    }
    
    /**
     * Check if table exists in database
     */
    private function tableExists(string $tableName): bool
    {
        return DB::connection()->getSchemaBuilder()->hasTable($tableName);
    }
}
