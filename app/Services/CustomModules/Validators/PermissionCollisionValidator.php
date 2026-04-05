<?php

namespace App\Services\CustomModules\Validators;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Schema;

class PermissionCollisionValidator
{
    /**
     * Validate that the module's permissions don't collide with existing ones
     */
    public function validate(string $filePath): array
    {
        try {
            $manifest = $this->extractManifest($filePath);
            $newPermissions = $manifest['permissions'] ?? [];
            $moduleName = $manifest['name'] ?? 'Unknown';
            
            if (empty($newPermissions)) {
                return ['valid' => true, 'issues' => [], 'permissions_count' => 0];
            }
            
            $issues = [];
            $checked = 0;
            
            foreach ($newPermissions as $perm) {
                $checked++;
                $permKey = $perm['key'] ?? null;
                
                if (!$permKey) {
                    $issues[] = [
                        'severity' => 'warning',
                        'code' => 'MALFORMED_PERMISSION',
                        'message' => 'Permission entry missing "key" field',
                    ];
                    continue;
                }
                
                // Check if permission already exists from a DIFFERENT module
                $keyColumn = Schema::hasColumn('permissions', 'permission_key') ? 'permission_key' : 'name';
                $existing = DB::table('permissions')
                    ->where($keyColumn, $permKey)
                    ->first();
                
                if ($existing && $existing->module !== $moduleName) {
                    $issues[] = [
                        'severity' => 'error',
                        'code' => 'PERMISSION_COLLISION',
                        'message' => "Permission key '{$permKey}' already exists in module '{$existing->module}'",
                        'conflicting_key' => $permKey,
                        'existing_module' => $existing->module,
                        'new_module' => $moduleName,
                    ];
                }
            }
            
            $hasErrors = !empty(array_filter($issues, fn($i) => $i['severity'] === 'error'));
            
            return [
                'valid' => !$hasErrors,
                'issues' => $issues,
                'permissions_count' => $checked,
                'collisions_found' => count(array_filter($issues, fn($i) => $i['code'] === 'PERMISSION_COLLISION')),
            ];
            
        } catch (\Exception $e) {
            return [
                'valid' => false,
                'issues' => [[
                    'severity' => 'error',
                    'code' => 'VALIDATION_ERROR',
                    'message' => "Permission validation failed: {$e->getMessage()}",
                ]],
                'permissions_count' => 0,
            ];
        }
    }
    
    /**
     * Extract and parse module.json from ZIP file
     */
    private function extractManifest(string $filePath): array
    {
        $tempDir = storage_path('temp/validator_' . uniqid());
        File::ensureDirectoryExists($tempDir);
        
        try {
            $zip = new \ZipArchive();
            if ($zip->open($filePath) !== true) {
                throw new \Exception('Cannot open ZIP file');
            }
            
            // Find module.json
            $manifestPath = null;
            for ($i = 0; $i < $zip->numFiles; $i++) {
                $name = $zip->getNameIndex($i);
                if (str_ends_with($name, 'module.json')) {
                    $manifestPath = $name;
                    break;
                }
            }
            
            if (!$manifestPath) {
                return []; // No manifest, will be caught by other validators
            }
            
            $content = $zip->getFromName($manifestPath);
            $zip->close();
            
            return json_decode($content, true) ?? [];
            
        } finally {
            File::deleteDirectory($tempDir);
        }
    }
}
