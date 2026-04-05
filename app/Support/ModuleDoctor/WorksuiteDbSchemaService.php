<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\Schema;

class WorksuiteDbSchemaService
{
    public function inspect(): array
    {
        $tables = [
            'packages',
            'permissions',
            'permission_role',
            'module_settings',
            'company_module_settings',
            'extensions',
            'roles',
            'users',
        ];

        $catalog = [];
        foreach ($tables as $table) {
            $catalog[$table] = [
                'exists' => Schema::hasTable($table),
                'columns' => Schema::hasTable($table) ? Schema::getColumnListing($table) : [],
            ];
        }

        return [
            'status' => 'pass',
            'detail' => 'Captured Worksuite database schema truth snapshot for menu visibility decisions.',
            'tables' => $catalog,
        ];
    }
}
