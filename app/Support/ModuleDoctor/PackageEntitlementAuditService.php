<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

class PackageEntitlementAuditService
{
    public function inspect(string $moduleName): array
    {
        if (!Schema::hasTable('packages')) {
            return [
                'status' => 'warn',
                'detail' => 'packages table not found for entitlement audit.',
                'packages' => [],
            ];
        }

        $rows = DB::table('packages')->select('id', 'package', 'module_in_package')->get();
        $matches = [];
        foreach ($rows as $row) {
            $modules = json_decode($row->module_in_package ?: '[]', true) ?: [];
            if (in_array($moduleName, $modules, true)) {
                $matches[] = [
                    'id' => $row->id,
                    'package' => $row->package,
                ];
            }
        }

        return [
            'status' => count($matches) ? 'pass' : 'warn',
            'detail' => count($matches)
                ? 'Module is granted in ' . count($matches) . ' package(s).'
                : 'Module is not currently attached to any package module_in_package list.',
            'packages' => $matches,
        ];
    }
}
