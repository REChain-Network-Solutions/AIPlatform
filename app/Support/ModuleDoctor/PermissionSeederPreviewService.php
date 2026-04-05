<?php

namespace App\Support\ModuleDoctor;

class PermissionSeederPreviewService
{
    public function inspect(array $permissionKeys = [], array $seeders = []): array
    {
        $preview = collect($permissionKeys)->map(function ($key) use ($seeders) {
            return [
                'permission_key' => $key,
                'seeded_by_module_seeder' => collect($seeders)->contains(fn ($path) => str_contains(strtolower($path), strtolower(str_replace('_', '', $key)))),
            ];
        })->all();

        return [
            'status' => count($preview) ? 'pass' : 'warn',
            'detail' => count($preview) ? 'Prepared permission seeder preview for inferred keys.' : 'No inferred keys available for seeder preview.',
            'preview' => $preview,
        ];
    }
}
