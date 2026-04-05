<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

class CompanyModuleSettingAuditService
{
    public function inspect(string $moduleName): array
    {
        $hasTable = Schema::hasTable('module_settings');
        if (!$hasTable) {
            return [
                'status' => 'warn',
                'detail' => 'module_settings table not found.',
                'rows' => [],
            ];
        }

        $query = DB::table('module_settings');
        if (Schema::hasColumn('module_settings', 'module_name')) {
            $query->where('module_name', $moduleName);
        } elseif (Schema::hasColumn('module_settings', 'module')) {
            $query->where('module', $moduleName);
        } else {
            return [
                'status' => 'warn',
                'detail' => 'module_settings table exists, but no module_name/module column was found.',
                'rows' => [],
            ];
        }

        $rows = $query->get()->map(fn ($row) => (array) $row)->all();

        return [
            'status' => count($rows) ? 'pass' : 'warn',
            'detail' => count($rows)
                ? 'module_settings already contains rows for this module.'
                : 'No module_settings rows were found for this module.',
            'rows' => $rows,
        ];
    }
}
