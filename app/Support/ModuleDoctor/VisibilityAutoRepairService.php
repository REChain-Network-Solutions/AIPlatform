<?php

namespace App\Support\ModuleDoctor;

class VisibilityAutoRepairService
{
    public function suggestions(array $visibilityReport): array
    {
        $repairs = [];
        if (!empty($visibilityReport['permissions']['db_audit']['missing'] ?? [])) {
            $repairs[] = [
                'title' => 'Seed missing permission keys',
                'detail' => 'Create the inferred permission keys before enabling tenant visibility.',
                'mode' => 'safe',
            ];
        }
        if (empty($visibilityReport['permissions']['role_matrix']['matrix'] ?? [])) {
            $repairs[] = [
                'title' => 'Attach permissions to role pivots',
                'detail' => 'Insert default permission_role mappings for admin-visible access.',
                'mode' => 'confirm',
            ];
        }
        if (empty($visibilityReport['package_bridge']['db_entitlements']['packages'] ?? [])) {
            $repairs[] = [
                'title' => 'Attach module to package entitlements',
                'detail' => 'Add the module to one or more packages so tenant users can receive it.',
                'mode' => 'safe',
            ];
        }
        if (empty($visibilityReport['package_bridge']['company_module_settings']['rows'] ?? [])) {
            $repairs[] = [
                'title' => 'Seed module_settings rows',
                'detail' => 'Insert company_module_settings rows for tenant companies after install.',
                'mode' => 'safe',
            ];
        }
        if (!empty($visibilityReport['route_name_mismatches']['items'] ?? [])) {
            $repairs[] = [
                'title' => 'Review sidebar route names',
                'detail' => 'Fix or rewrite named route references before expecting menu visibility.',
                'mode' => 'confirm',
            ];
        }
        foreach (app(SidebarPermissionPatternRepairService::class)->suggestions($visibilityReport['sidebar'] ?? [], $visibilityReport['permissions']['permission_keys'] ?? []) as $repair) {
            $repairs[] = $repair;
        }

        return $repairs;
    }
}
