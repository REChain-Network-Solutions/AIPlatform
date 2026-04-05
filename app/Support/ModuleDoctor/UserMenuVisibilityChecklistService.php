<?php

namespace App\Support\ModuleDoctor;

class UserMenuVisibilityChecklistService
{
    public function build(array $visibility): array
    {
        return [
            ['label' => 'Sidebar file detected', 'status' => !empty($visibility['sidebar']['files']) ? 'pass' : 'warn'],
            ['label' => 'Sidebar include chain detected', 'status' => !empty($visibility['sidebar']['inclusion']['core_sidebar_hits']) ? 'pass' : 'warn'],
            ['label' => 'Sidebar route names resolved', 'status' => empty($visibility['sidebar']['route_resolution']['missing_route_names'] ?? []) ? 'pass' : 'warn'],
            ['label' => 'Permission keys inferred', 'status' => !empty($visibility['permissions']['permission_keys']) ? 'pass' : 'warn'],
            ['label' => 'Permissions seeded', 'status' => empty($visibility['permissions']['db_audit']['missing'] ?? []) ? 'pass' : 'warn'],
            ['label' => 'Package entitlement rows', 'status' => !empty($visibility['package_bridge']['db_entitlements']['packages'] ?? []) ? 'pass' : 'warn'],
            ['label' => 'Tenant module settings rows', 'status' => !empty($visibility['package_bridge']['company_module_settings']['rows'] ?? []) ? 'pass' : 'warn'],
        ];
    }
}
