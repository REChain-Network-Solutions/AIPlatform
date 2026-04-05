<?php

namespace App\Support\ModuleDoctor;

class SidebarPermissionPatternRepairService
{
    public function suggestions(array $sidebarAudit, array $permissionKeys = []): array
    {
        $repairs = [];
        foreach (($sidebarAudit['unsupported_patterns'] ?? []) as $pattern) {
            $repairs[] = [
                'title' => 'Rewrite sidebar permission guard',
                'detail' => 'Replace unsupported pattern ' . $pattern . ' with Worksuite-safe permission checks.',
                'mode' => 'confirm',
            ];
        }

        if (empty($permissionKeys) && !empty($sidebarAudit['menu_ready'])) {
            $repairs[] = [
                'title' => 'Infer sidebar permission keys',
                'detail' => 'Sidebar exists but no permission keys were inferred. Add explicit Worksuite permission guards.',
                'mode' => 'confirm',
            ];
        }

        return $repairs;
    }
}
