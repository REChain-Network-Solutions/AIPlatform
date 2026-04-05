<?php

namespace App\Support\ModuleDoctor;

class WorksuiteProfileService
{
    public function profile(): array
    {
        return [
            'workspace' => 'worksuite',
            'tenant_gate' => 'company + package + module_settings',
            'preferred_permission_api' => "user()->permission('key') == 'all'",
            'protected_paths' => app(ProtectedPathService::class)->all(),
        ];
    }
}
