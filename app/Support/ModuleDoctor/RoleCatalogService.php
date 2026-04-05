<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class RoleCatalogService
{
    public function inspect(): array
    {
        if (!Schema::hasTable('roles')) {
            return ['status' => 'warn', 'detail' => 'roles table not found.', 'roles' => []];
        }

        $roles = DB::table('roles')->select('id', 'name', 'display_name')->orderBy('id')->get()->map(function ($row) {
            return [
                'id' => (int) $row->id,
                'name' => $row->name,
                'label' => $row->display_name ?: $row->name,
            ];
        })->all();

        return [
            'status' => count($roles) ? 'pass' : 'warn',
            'detail' => count($roles) ? count($roles) . ' role(s) discovered.' : 'No roles were found in the roles table.',
            'roles' => $roles,
        ];
    }
}
