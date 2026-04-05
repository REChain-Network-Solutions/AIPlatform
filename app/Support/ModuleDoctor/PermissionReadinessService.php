<?php

namespace App\Support\ModuleDoctor;

class PermissionReadinessService
{
    public function inspect(array $module): array
    {
        $modulePath = $module['path'] ?? '';
        $extracted = app(PermissionUsageExtractorService::class)->extract($modulePath);
        $keys = $extracted['permission_keys'] ?? [];
        $roleMatrix = app(PermissionPivotMatrixService::class)->inspect($keys);
        $matrix = $roleMatrix['matrix'] ?? [];

        return [
            'keys' => $keys,
            'permission_keys' => $keys,
            'missing_keys' => array_values(array_filter($keys, fn ($key) => empty($matrix[$key]))),
            'unmapped_keys' => array_values(array_filter($keys, fn ($key) => empty($matrix[$key]))),
            'ready' => !empty($keys),
            'matrix' => $roleMatrix,
        ];
    }
}
