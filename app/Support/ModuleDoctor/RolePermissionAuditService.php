<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

class RolePermissionAuditService
{
    public function inspect(array $permissionKeys): array
    {
        if (!function_exists('app') || !Schema::hasTable('permissions')) {
            return [
                'status' => 'warn',
                'detail' => 'Permission tables were not available during analysis.',
                'seeded' => [],
                'missing' => $permissionKeys,
                'role_links' => [],
            ];
        }

        $seeded = DB::table('permissions')->whereIn('name', $permissionKeys)->pluck('name')->all();
        $missing = array_values(array_diff($permissionKeys, $seeded));
        $roleLinks = [];
        if (Schema::hasTable('permission_role') && Schema::hasTable('permissions')) {
            $roleLinks = DB::table('permission_role')
                ->join('permissions', 'permissions.id', '=', 'permission_role.permission_id')
                ->whereIn('permissions.name', $seeded)
                ->select('permissions.name', 'permission_role.role_id')
                ->get()
                ->groupBy('name')
                ->map(fn ($rows) => $rows->pluck('role_id')->unique()->values()->all())
                ->all();
        }

        return [
            'status' => count($missing) ? 'warn' : 'pass',
            'detail' => count($permissionKeys)
                ? (count($missing)
                    ? count($missing) . ' inferred permission key(s) are not currently seeded in the database.'
                    : 'All inferred permission keys already exist in the database.')
                : 'No inferred permission keys to audit.',
            'seeded' => array_values($seeded),
            'missing' => $missing,
            'role_links' => $roleLinks,
        ];
    }
}
