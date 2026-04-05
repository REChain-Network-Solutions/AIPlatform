<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class CompanyModuleSettingsTruthService
{
    public function inspect(string $moduleName): array
    {
        if (!Schema::hasTable('company_module_settings')) {
            return ['status' => 'warn', 'detail' => 'company_module_settings table not found.', 'rows' => []];
        }

        $rows = DB::table('company_module_settings')->where('module_name', $moduleName)->limit(50)->get()->map(fn ($row) => (array) $row)->all();

        return [
            'status' => count($rows) ? 'pass' : 'warn',
            'detail' => count($rows) ? count($rows) . ' company_module_settings row(s) found.' : 'No company_module_settings rows exist for this module.',
            'rows' => $rows,
        ];
    }
}
