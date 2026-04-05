<?php

namespace App\Services\CustomModules\Validators;

use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\Route;

class RouteCollisionValidator
{
    /**
     * Validate that the module's routes don't collide with existing ones
     */
    public function validate(string $filePath): array
    {
        try {
            $manifest = $this->extractManifest($filePath);
            $newRoutes = $manifest['routes'] ?? [];
            $moduleName = $manifest['name'] ?? 'Unknown';
            
            if (empty($newRoutes)) {
                return ['valid' => true, 'issues' => [], 'routes_count' => 0];
            }
            
            $issues = [];
            $checked = 0;
            $existingRoutes = $this->getExistingRoutes();
            
            foreach ($newRoutes as $route) {
                $checked++;
                $uri = $route['uri'] ?? null;
                $method = strtoupper($route['method'] ?? 'GET');
                
                if (!$uri) {
                    $issues[] = [
                        'severity' => 'warning',
                        'code' => 'MALFORMED_ROUTE',
                        'message' => 'Route entry missing "uri" field',
                    ];
                    continue;
                }
                
                // Normalize URI for comparison
                $normalizedUri = trim($uri, '/');
                
                // Check if route already exists
                foreach ($existingRoutes as $existing) {
                    $existingUri = trim($existing['uri'], '/');
                    
                    if ($normalizedUri === $existingUri) {
                        // Check if method matches or if it's a conflict
                        if (in_array($method, (array) $existing['methods'])) {
                            $issues[] = [
                                'severity' => 'error',
                                'code' => 'ROUTE_COLLISION',
                                'message' => "Route {$method} /{$normalizedUri} already registered",
                                'conflicting_uri' => $uri,
                                'conflicting_method' => $method,
                                'existing_uri' => $existing['uri'],
                                'existing_methods' => $existing['methods'],
                            ];
                            break;
                        }
                    }
                }
            }
            
            $hasErrors = !empty(array_filter($issues, fn($i) => $i['severity'] === 'error'));
            
            return [
                'valid' => !$hasErrors,
                'issues' => $issues,
                'routes_count' => $checked,
                'collisions_found' => count(array_filter($issues, fn($i) => $i['code'] === 'ROUTE_COLLISION')),
            ];
            
        } catch (\Exception $e) {
            return [
                'valid' => false,
                'issues' => [[
                    'severity' => 'error',
                    'code' => 'VALIDATION_ERROR',
                    'message' => "Route validation failed: {$e->getMessage()}",
                ]],
                'routes_count' => 0,
            ];
        }
    }
    
    /**
     * Extract routes from module.json manifest
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
                return [];
            }
            
            $content = $zip->getFromName($manifestPath);
            $zip->close();
            
            return json_decode($content, true) ?? [];
            
        } finally {
            File::deleteDirectory($tempDir);
        }
    }
    
    /**
     * Get all currently registered routes in the application
     */
    private function getExistingRoutes(): array
    {
        $routes = [];
        $routeCollection = Route::getRoutes();
        
        foreach ($routeCollection as $route) {
            $routes[] = [
                'uri' => $route->uri(),
                'methods' => array_diff($route->methods(), ['HEAD']), // Exclude HEAD
                'name' => $route->getName(),
            ];
        }
        
        return $routes;
    }
}
