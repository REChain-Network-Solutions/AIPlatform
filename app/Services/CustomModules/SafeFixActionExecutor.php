<?php

namespace App\Services\CustomModules;

use App\Services\CustomModules\Actions\{
    QueuePermissionRepairAction,
    QueuePackageBridgeRepairAction,
    QueueCompanyModuleSettingsRepairAction,
    QueueCacheClearAction,
};

class SafeFixActionExecutor
{
    protected array $actions = [];
    
    public function __construct(
        protected QueuePermissionRepairAction $permissionAction = new QueuePermissionRepairAction(),
        protected QueuePackageBridgeRepairAction $packageAction = new QueuePackageBridgeRepairAction(),
        protected QueueCompanyModuleSettingsRepairAction $companyAction = new QueueCompanyModuleSettingsRepairAction(),
        protected QueueCacheClearAction $cacheAction = new QueueCacheClearAction(),
    ) {
        $this->registerActions();
    }
    
    /**
     * Execute all selected repairs in sequence
     */
    public function executeAll(string $moduleName, array $selectedRepairs = []): array
    {
        $executed = [];
        $failed = [];
        $skipped = [];
        
        // Determine which repairs to run
        $toRun = [
            'repair_permissions' => $selectedRepairs['repair_permissions'] ?? true,
            'repair_package_bridge' => $selectedRepairs['repair_package_bridge'] ?? true,
            'repair_company_module_settings' => $selectedRepairs['repair_company_module_settings'] ?? true,
            'clear_caches' => $selectedRepairs['clear_caches'] ?? true,
        ];
        
        // Execute in order
        foreach ($toRun as $actionName => $shouldRun) {
            if (!$shouldRun) {
                $skipped[$actionName] = 'Skipped by user request';
                continue;
            }
            
            if (!isset($this->actions[$actionName])) {
                $failed[$actionName] = 'Action not registered';
                continue;
            }
            
            $action = $this->actions[$actionName];
            
            // Check if action can execute
            if (!$action->canExecute($moduleName)) {
                $skipped[$actionName] = 'Cannot execute (preconditions not met)';
                continue;
            }
            
            try {
                $result = $action->execute($moduleName);
                
                if ($result['success'] ?? false) {
                    $executed[$actionName] = $result;
                } else {
                    $failed[$actionName] = $result;
                }
            } catch (\Exception $e) {
                $failed[$actionName] = [
                    'success' => false,
                    'error' => $e->getMessage(),
                    'message' => "Exception during execution: {$e->getMessage()}",
                ];
            }
        }
        
        return [
            'all_success' => empty($failed),
            'executed' => $executed,
            'failed' => $failed,
            'skipped' => $skipped,
            'summary' => [
                'total_executed' => count($executed),
                'total_failed' => count($failed),
                'total_skipped' => count($skipped),
            ],
        ];
    }
    
    /**
     * Rollback executed repairs in reverse order
     */
    public function rollbackRepairs(string $moduleName, array $executedRepairs): array
    {
        $results = [];
        
        // Reverse order (undo in opposite sequence)
        $actionNames = array_reverse(array_keys($executedRepairs));
        
        foreach ($actionNames as $actionName) {
            if (!isset($this->actions[$actionName])) {
                $results[$actionName] = ['success' => false, 'message' => 'Action not registered'];
                continue;
            }
            
            try {
                $action = $this->actions[$actionName];
                $result = $action->rollback($moduleName);
                $results[$actionName] = $result;
            } catch (\Exception $e) {
                $results[$actionName] = [
                    'success' => false,
                    'error' => $e->getMessage(),
                    'message' => "Rollback failed: {$e->getMessage()}",
                ];
            }
        }
        
        $allSuccess = empty(array_filter(
            $results,
            fn($r) => ($r['success'] ?? false) === false
        ));
        
        return [
            'all_success' => $allSuccess,
            'results' => $results,
        ];
    }
    
    /**
     * Register all available actions
     */
    private function registerActions(): void
    {
        $this->actions = [
            'repair_permissions' => $this->permissionAction,
            'repair_package_bridge' => $this->packageAction,
            'repair_company_module_settings' => $this->companyAction,
            'clear_caches' => $this->cacheAction,
        ];
    }
    
    /**
     * Get action descriptions for planning phase
     */
    public function getActionDescriptions(): array
    {
        $descriptions = [];
        
        foreach ($this->actions as $name => $action) {
            $descriptions[$name] = $action->describe();
        }
        
        return $descriptions;
    }
}
