<?php

namespace App\Support\ModuleDoctor;

class KnownIssueRegistry
{
    public function all(): array
    {
        return [
            'foreign_permission_api' => 'Module sidebar or controller uses non-Worksuite permission checks.',
            'missing_sidebar' => 'Routes exist but no sidebar/menu surface was detected.',
            'missing_route_registration' => 'Named routes referenced by the module are not registered at runtime.',
            'missing_permission_seed' => 'Permission keys inferred from code are not seeded or mapped to roles.',
            'missing_company_module_setting' => 'Module is enabled globally but not for current tenant/company.',
            'missing_package_bridge' => 'Tenant package does not grant access to this module.',
            'missing_parent_menu' => 'Menu row has no valid parent workspace anchor.',
            'bad_index_route' => 'Primary menu route does not end in .index or cannot resolve.',
            'provider_boot_failure' => 'Service provider is absent or route boot path is invalid.',
            'hidden_by_workspace' => 'Sidebar exists but is not included in Worksuite active workspace chain.',
        ];
    }
}
