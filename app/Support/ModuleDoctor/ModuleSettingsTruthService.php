<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class ModuleSettingsTruthService
{
    public function inspect(string $moduleName): array
    {
        if (!Schema::hasTable('module_settings')) {
            return ['status' => 'warn', 'detail' => 'module_settings table not found.', 'rows' => []];
        }

        $rows = DB::table('module_settings')->where('module_name', $moduleName)->get()->map(fn ($row) => (array) $row)->all();

        return [
            'status' => count($rows) ? 'pass' : 'warn',
            'detail' => count($rows) ? count($rows) . ' module_settings row(s) found.' : 'No module_settings rows exist for this module.',
            'rows' => $rows,
        ];
    }
}
