<?php

namespace Modules\WorkOrders\Database\Seeders;

use Illuminate\Database\Seeder;
use Spatie\Permission\Models\Role;
use Spatie\Permission\Models\Permission;
use App\Models\User;

class WorkOrdersPermissionSeeder extends Seeder
{
    public function run(): void
    {
        $perms = [
            'workorders.view','workorders.create','workorders.update','workorders.delete',
            'workorders.types.manage','workorders.requests.manage',
            'workorders.appointments.manage',
            'workorders.tasks.manage',
            'workorders.parts.manage',
        ];

        foreach ($perms as $p) { Permission::firstOrCreate(['name' => $p]); }

        $admin = Role::firstOrCreate(['name' => 'admin']);
        $tech  = Role::firstOrCreate(['name' => 'technician']);
        $viewer= Role::firstOrCreate(['name' => 'viewer']);

        $admin->syncPermissions(Permission::all());
        $tech->syncPermissions([
            'workorders.view','workorders.update',
            'workorders.appointments.manage',
            'workorders.tasks.manage','workorders.parts.manage',
        ]);
        $viewer->syncPermissions(['workorders.view']);

        if ($user = User::first()) { $user->assignRole($admin); }
    }
}
