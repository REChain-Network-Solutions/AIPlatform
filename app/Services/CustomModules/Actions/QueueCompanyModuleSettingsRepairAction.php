<?php

namespace App\Services\CustomModules\Actions;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;

class QueueCompanyModuleSettingsRepairAction
{
    /**
     * Describe this action
     */
    public function describe(): array
    {
        return [
            'name' => 'repair_company_module_settings',
            'title' => 'Initialize Company Module Settings',
            'description' => 'Create module_settings records for each company',
            'estimated_duration' => '2-5 seconds',
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
     * Execute company module settings repair
     */
    public function execute(string $moduleName): array
    {
        try {
            // Check if module_settings table exists
            if (!$this->tableExists('module_settings')) {
                return [
                    'success' => true,
                    'message' => 'module_settings table not found - skipping',
                    'skipped' => true,
                ];
            }
            
            // Check if modules table exists to get module_id
            if (!$this->tableExists('modules')) {
                return [
                    'success' => true,
                    'message' => 'modules table not found - skipping',
                    'skipped' => true,
                ];
            }
            
            // Get module ID
            $module = DB::table('modules')
                ->where('module_name', $moduleName)
                ->first();
            
            if (!$module) {
                return [
                    'success' => false,
                    'error' => "Module {$moduleName} not found in modules table",
                    'message' => "Cannot create module settings without module registration",
                ];
            }
            
            // Get all companies
            $companies = DB::table('companies')->pluck('id')->toArray();
            
            $created = 0;
            $errors = [];
            
            foreach ($companies as $companyId) {
                try {
                    // Check if settings already exist
                    $exists = DB::table('module_settings')
                        ->where('module_id', $module->id)
                        ->where('company_id', $companyId)
                        ->exists();
                    
                    if (!$exists) {
                        DB::table('module_settings')->insert([
                            'module_id' => $module->id,
                            'company_id' => $companyId,
                            'setting_key' => 'enabled',
                            'setting_value' => json_encode(['status' => 'active']),
                            'created_at' => now(),
                            'updated_at' => now(),
                        ]);
                        $created++;
                    }
                } catch (\Exception $e) {
                    $errors[] = "Company {$companyId}: {$e->getMessage()}";
                }
            }
            
            return [
                'success' => empty($errors),
                'message' => "Created module settings for {$created} companies",
                'created' => $created,
                'errors' => $errors,
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'message' => "Company module settings repair failed: {$e->getMessage()}",
            ];
        }
    }
    
    /**
     * Rollback company module settings
     */
    public function rollback(string $moduleName): array
    {
        try {
            if (!$this->tableExists('module_settings') || !$this->tableExists('modules')) {
                return [
                    'success' => true,
                    'message' => 'Required tables not found - nothing to rollback',
                ];
            }
            
            $module = DB::table('modules')
                ->where('module_name', $moduleName)
                ->first();
            
            if (!$module) {
                return [
                    'success' => true,
                    'message' => 'Module not found - nothing to rollback',
                ];
            }
            
            $deleted = DB::table('module_settings')
                ->where('module_id', $module->id)
                ->delete();
            
            return [
                'success' => true,
                'message' => "Removed module settings from {$deleted} companies",
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
     * Check if table exists
     */
    private function tableExists(string $tableName): bool
    {
        return DB::connection()->getSchemaBuilder()->hasTable($tableName);
    }
}
