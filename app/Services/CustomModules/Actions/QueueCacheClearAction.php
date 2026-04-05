<?php

namespace App\Services\CustomModules\Actions;

use Illuminate\Support\Facades\Artisan;

class QueueCacheClearAction
{
    /**
     * Describe this action
     */
    public function describe(): array
    {
        return [
            'name' => 'clear_caches',
            'title' => 'Clear Application Caches',
            'description' => 'Clear route, view, and config caches to reload module',
            'estimated_duration' => '1-2 seconds',
            'reversible' => false,
            'critical' => true,
        ];
    }
    
    /**
     * Check if action can execute
     */
    public function canExecute(string $moduleName): bool
    {
        return true; // Can always clear caches
    }
    
    /**
     * Execute cache clearing
     */
    public function execute(string $moduleName): array
    {
        try {
            $commands = [
                'cache:clear' => 'Clear application cache',
                'route:clear' => 'Clear route cache',
                'view:clear' => 'Clear view cache',
                'config:clear' => 'Clear config cache',
            ];
            
            $results = [];
            $errors = [];
            
            foreach ($commands as $command => $description) {
                try {
                    Artisan::call($command);
                    $results[$command] = 'success';
                } catch (\Exception $e) {
                    $errors[$command] = $e->getMessage();
                }
            }
            
            return [
                'success' => empty($errors),
                'message' => 'Application caches cleared',
                'caches_cleared' => count($results),
                'errors' => $errors,
            ];
            
        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage(),
                'message' => "Cache clear failed: {$e->getMessage()}",
            ];
        }
    }
    
    /**
     * Rollback (no-op for cache clearing)
     */
    public function rollback(string $moduleName): array
    {
        // Cache clear is not reversible
        return [
            'success' => true,
            'message' => 'Cache clear cannot be rolled back (not needed)',
        ];
    }
}
