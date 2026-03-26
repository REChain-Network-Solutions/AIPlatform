<?php

namespace Modules\ComplianceIQ\Database\Seeders;

use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class CompliancePermissionSeeder extends Seeder
{
    public function run(): void
    {
        $perms = [
            'compliance.view','compliance.create','compliance.update',
            'compliance.signoff','compliance.export','compliance.logs.view'
        ];

        foreach ($perms as $p) {
            DB::table('permissions')->updateOrInsert(['name' => $p], ['guard_name' => 'web']);
        }

        $roleMap = [
            'Admin' => $perms,
            'Compliance Officer' => $perms,
            'Auditor' => ['compliance.view','compliance.logs.view'],
        ];

        foreach ($roleMap as $roleName => $permNames) {
            $role = DB::table('roles')->where('name', $roleName)->first();
            if (!$role) continue;
            foreach ($permNames as $perm) {
                $permId = DB::table('permissions')->where('name', $perm)->value('id');
                if ($permId) {
                    DB::table('role_has_permissions')->updateOrInsert([
                        'role_id' => $role->id,
                        'permission_id' => $permId,
                    ], []);
                }
            }
        }
    }
}
