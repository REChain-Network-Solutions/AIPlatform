<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

class CompanyPermissionScopeAuditService
{
    public function inspect(array $context, array $permissionKeys = []): array
    {
        $companyId = null;
        try {
            if (function_exists('company') && company()) {
                $companyId = company()->id ?? null;
            }
        } catch (\Throwable $e) {
            $companyId = null;
        }

        $hasCustom = Schema::hasTable('custom_module_permissions');
        $companyRows = [];
        if ($hasCustom && $companyId && count($permissionKeys)) {
            $companyRows = DB::table('custom_module_permissions')
                ->where('company_id', $companyId)
                ->whereIn('permission_name', $permissionKeys)
                ->pluck('permission_name')
                ->all();
        }

        return [
            'status' => $companyId ? 'pass' : 'warn',
            'detail' => $companyId
                ? 'Company scope was available for audit' . ($hasCustom ? ' and custom module permission table exists.' : ', but no custom_module_permissions table was found.')
                : 'No active company scope was available during analysis.',
            'company_id' => $companyId,
            'company_permission_rows' => array_values($companyRows),
            'has_custom_permission_table' => $hasCustom,
        ];
    }
}
