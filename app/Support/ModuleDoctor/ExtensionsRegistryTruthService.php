<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class ExtensionsRegistryTruthService
{
    public function inspect(string $moduleName): array
    {
        if (!Schema::hasTable('extensions')) {
            return ['status' => 'warn', 'detail' => 'extensions table not found.', 'rows' => []];
        }

        $rows = DB::table('extensions')
            ->where('name', $moduleName)
            ->orWhere('module_name', $moduleName)
            ->get()
            ->map(fn ($row) => (array) $row)
            ->all();

        return [
            'status' => count($rows) ? 'pass' : 'warn',
            'detail' => count($rows) ? 'Extension registry rows found for this module.' : 'No extension registry row was found for this module.',
            'rows' => $rows,
        ];
    }
}
