<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class PermissionPivotMatrixService
{
    public function inspect(array $permissionKeys = []): array
    {
        if (!Schema::hasTable('permission_role') || !Schema::hasTable('permissions')) {
            return ['status' => 'warn', 'detail' => 'permission_role or permissions table missing.', 'matrix' => []];
        }

        if (empty($permissionKeys)) {
            return ['status' => 'warn', 'detail' => 'No inferred permission keys were provided.', 'matrix' => []];
        }

        $rows = DB::table('permission_role')
            ->join('permissions', 'permissions.id', '=', 'permission_role.permission_id')
            ->whereIn('permissions.name', $permissionKeys)
            ->select('permissions.name as permission_name', 'permission_role.role_id')
            ->orderBy('permission_name')
            ->get();

        $matrix = $rows->groupBy('permission_name')->map(fn ($group) => $group->pluck('role_id')->map(fn ($id) => (int) $id)->unique()->values()->all())->all();

        return [
            'status' => count($matrix) ? 'pass' : 'warn',
            'detail' => count($matrix) ? 'Built permission-to-role matrix for inferred permission keys.' : 'No role mappings were found for inferred permission keys.',
            'matrix' => $matrix,
        ];
    }
}
