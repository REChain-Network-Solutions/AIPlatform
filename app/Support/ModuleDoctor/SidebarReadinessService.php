<?php

namespace App\Support\ModuleDoctor;

class SidebarReadinessService
{
    public function inspect(array $module): array
    {
        $files = $module['files'] ?? [];
        $sidebarFiles = array_values(array_filter($files, fn ($file) => str_contains($file, 'sidebar') && str_ends_with($file, '.blade.php')));
        $usesForeignPermissionApi = (new \Illuminate\Support\Collection($files))->contains(fn ($file) => str_contains($file, 'isAbleTo') || str_contains($file, '@permission'));
        $hasWorksuiteGate = (new \Illuminate\Support\Collection($files))->contains(fn ($file) => str_contains($file, "user()->permission(") || str_contains($file, "module_enabled("));

        return [
            'sidebar_files' => $sidebarFiles,
            'has_sidebar' => count($sidebarFiles) > 0,
            'uses_foreign_permission_api' => $usesForeignPermissionApi,
            'has_worksuite_gate' => $hasWorksuiteGate,
            'ready' => count($sidebarFiles) > 0 && !$usesForeignPermissionApi,
        ];
    }
}
