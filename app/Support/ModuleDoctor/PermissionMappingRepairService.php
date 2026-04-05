<?php

namespace App\Support\ModuleDoctor;

class PermissionMappingRepairService
{
    public function plan(array $dbAudit): array
    {
        return [
            'status' => empty($dbAudit['missing'] ?? []) ? 'pass' : 'warn',
            'detail' => empty($dbAudit['missing'] ?? [])
                ? 'No permission mapping repair is required.'
                : 'Seed missing permissions first, then map them to admin-visible roles.',
        ];
    }
}
