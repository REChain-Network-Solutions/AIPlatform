<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;
use Nwidart\Modules\Facades\Module;

class ModuleRegistryAuditService
{
    public function inspect(string $moduleName): array
    {
        $module = null;
        try {
            $module = Module::find($moduleName);
        } catch (\Throwable $e) {
            $module = null;
        }

        $extensionRows = [];
        if (Schema::hasTable('custom_modules')) {
            $extensionRows = DB::table('custom_modules')->where('module_name', $moduleName)->get()->map(fn ($row) => (array) $row)->all();
        }

        return [
            'status' => $module || count($extensionRows) ? 'pass' : 'warn',
            'detail' => $module
                ? 'Module is already known to the Nwidart registry.'
                : (count($extensionRows)
                    ? 'Custom module registry rows exist for this module.'
                    : 'No module registry evidence was found.'),
            'nwidart_registered' => (bool) $module,
            'custom_module_rows' => $extensionRows,
        ];
    }
}
