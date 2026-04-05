<?php

namespace App\Support\ModuleDoctor;

use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class SafeAutoFixExecutionService
{
    public function run(string $moduleName, array $analysis = [], string $attachMode = 'none', ?Collection $selectedPackageIds = null): array
    {
        $selectedPackageIds = $selectedPackageIds ?: collect();

        $results = [
            'status' => 'pass',
            'detail' => 'Safe autofix execution finished.',
            'applied' => [],
            'warnings' => [],
        ];

        $permissionKeys = collect($analysis['visibility_report']['permissions']['permission_keys'] ?? [])->filter()->unique()->values();
        if ($permissionKeys->isNotEmpty()) {
            $seed = $this->seedPermissions($permissionKeys);
            $results['applied'][] = $seed;

            $map = $this->mapPermissionsToAdminRoles($permissionKeys);
            $results['applied'][] = $map;
        }

        $packageFix = $this->attachModuleToPackages($moduleName, $attachMode, $selectedPackageIds);
        if ($packageFix) {
            $results['applied'][] = $packageFix;
        }

        $moduleSettings = $this->seedModuleSettings($moduleName);
        if ($moduleSettings) {
            $results['applied'][] = $moduleSettings;
        }

        $companySettings = $this->seedCompanyModuleSettings($moduleName);
        if ($companySettings) {
            $results['applied'][] = $companySettings;
        }

        try {
            Artisan::call('optimize:clear');
            $results['applied'][] = [
                'title' => 'Clear runtime caches',
                'detail' => 'Cleared config, route, view, and application caches after safe autofix.',
                'mode' => 'safe',
            ];
        }
        catch (\Throwable $e) {
            $results['warnings'][] = 'Cache clear failed: ' . $e->getMessage();
        }

        if (!empty($results['warnings'])) {
            $results['status'] = 'warn';
            $results['detail'] = 'Safe autofix applied with warnings.';
        }

        return $results;
    }

    private function seedPermissions(Collection $permissionKeys): array
    {
        if (!Schema::hasTable('permissions')) {
            return [
                'title' => 'Seed permission keys',
                'detail' => 'Skipped because permissions table was not found.',
                'mode' => 'safe',
            ];
        }

        $columns = Schema::getColumnListing('permissions');
        $created = 0;
        $now = now();

        foreach ($permissionKeys as $name) {
            $exists = DB::table('permissions')->where('name', $name)->exists();
            if ($exists) {
                continue;
            }

            $row = [];
            if (in_array('name', $columns, true)) {
                $row['name'] = $name;
            }
            if (in_array('display_name', $columns, true)) {
                $row['display_name'] = ucwords(str_replace(['_', '-'], ' ', $name));
            }
            if (in_array('display_name_hindi', $columns, true)) {
                $row['display_name_hindi'] = ucwords(str_replace(['_', '-'], ' ', $name));
            }
            if (in_array('permission_group_id', $columns, true) && Schema::hasTable('permission_groups')) {
                $groupId = DB::table('permission_groups')->orderBy('id')->value('id');
                if ($groupId) {
                    $row['permission_group_id'] = $groupId;
                }
            }
            if (in_array('is_custom', $columns, true)) {
                $row['is_custom'] = 1;
            }
            if (in_array('created_at', $columns, true)) {
                $row['created_at'] = $now;
            }
            if (in_array('updated_at', $columns, true)) {
                $row['updated_at'] = $now;
            }

            if (!empty($row)) {
                DB::table('permissions')->insert($row);
                $created++;
            }
        }

        return [
            'title' => 'Seed permission keys',
            'detail' => $created . ' missing permission key(s) inserted into permissions.',
            'mode' => 'safe',
        ];
    }

    private function mapPermissionsToAdminRoles(Collection $permissionKeys): array
    {
        if (!Schema::hasTable('roles') || !Schema::hasTable('permission_role') || !Schema::hasTable('permissions')) {
            return [
                'title' => 'Map permissions to admin roles',
                'detail' => 'Skipped because roles/permission_role/permissions tables were not all present.',
                'mode' => 'safe',
            ];
        }

        $roleColumns = Schema::getColumnListing('roles');
        $nameColumn = in_array('name', $roleColumns, true) ? 'name' : (in_array('display_name', $roleColumns, true) ? 'display_name' : null);
        if (!$nameColumn) {
            return [
                'title' => 'Map permissions to admin roles',
                'detail' => 'Skipped because roles table has no recognised role name column.',
                'mode' => 'safe',
            ];
        }

        $adminRoles = DB::table('roles')
            ->where(function ($query) use ($nameColumn) {
                $query->where($nameColumn, 'admin')
                    ->orWhere($nameColumn, 'administrator')
                    ->orWhere($nameColumn, 'superadmin')
                    ->orWhere($nameColumn, 'super-admin');
            })
            ->pluck('id');

        if ($adminRoles->isEmpty()) {
            return [
                'title' => 'Map permissions to admin roles',
                'detail' => 'Skipped because no admin-like roles were found.',
                'mode' => 'safe',
            ];
        }

        $permissionIds = DB::table('permissions')->whereIn('name', $permissionKeys->all())->pluck('id');
        $inserted = 0;
        $pivotColumns = Schema::getColumnListing('permission_role');
        $now = now();

        foreach ($adminRoles as $roleId) {
            foreach ($permissionIds as $permissionId) {
                $exists = DB::table('permission_role')->where('role_id', $roleId)->where('permission_id', $permissionId)->exists();
                if ($exists) {
                    continue;
                }
                $row = [
                    'role_id' => $roleId,
                    'permission_id' => $permissionId,
                ];
                if (in_array('created_at', $pivotColumns, true)) {
                    $row['created_at'] = $now;
                }
                if (in_array('updated_at', $pivotColumns, true)) {
                    $row['updated_at'] = $now;
                }
                DB::table('permission_role')->insert($row);
                $inserted++;
            }
        }

        return [
            'title' => 'Map permissions to admin roles',
            'detail' => $inserted . ' permission_role mapping(s) inserted for admin-visible roles.',
            'mode' => 'safe',
        ];
    }

    private function attachModuleToPackages(string $moduleName, string $attachMode, Collection $selectedPackageIds): ?array
    {
        if ($attachMode === 'none' || !Schema::hasTable('packages') || !Schema::hasColumn('packages', 'module_in_package')) {
            return null;
        }

        $query = DB::table('packages')->select('id', 'module_in_package');
        if ($attachMode === 'selected' && $selectedPackageIds->isNotEmpty()) {
            $query->whereIn('id', $selectedPackageIds->all());
        }

        $updated = 0;
        foreach ($query->get() as $package) {
            $modules = collect(json_decode($package->module_in_package ?: '[]', true))->filter()->values();
            if ($modules->contains($moduleName)) {
                continue;
            }
            $modules->push($moduleName);
            DB::table('packages')->where('id', $package->id)->update(['module_in_package' => $modules->values()->toJson()]);
            $updated++;
        }

        return [
            'title' => 'Attach module to packages',
            'detail' => $updated . ' package entitlement row(s) updated for this module.',
            'mode' => 'safe',
        ];
    }

    private function seedModuleSettings(string $moduleName): ?array
    {
        if (!Schema::hasTable('module_settings')) {
            return null;
        }

        $columns = Schema::getColumnListing('module_settings');
        if (!in_array('module_name', $columns, true) || !in_array('type', $columns, true)) {
            return [
                'title' => 'Seed module_settings',
                'detail' => 'Skipped because module_settings schema is not compatible.',
                'mode' => 'safe',
            ];
        }

        $types = ['admin', 'employee', 'client'];
        $companyIds = collect();
        if (Schema::hasTable('companies') && Schema::hasColumn('module_settings', 'company_id')) {
            $companyIds = DB::table('companies')->pluck('id');
        }
        if ($companyIds->isEmpty()) {
            $companyIds = collect([null]);
        }

        $inserted = 0;
        $now = now();
        foreach ($companyIds as $companyId) {
            foreach ($types as $type) {
                $query = DB::table('module_settings')->where('module_name', $moduleName)->where('type', $type);
                if (Schema::hasColumn('module_settings', 'company_id')) {
                    $query = $companyId === null ? $query->whereNull('company_id') : $query->where('company_id', $companyId);
                }
                if ($query->exists()) {
                    continue;
                }

                $row = [
                    'module_name' => $moduleName,
                    'type' => $type,
                ];
                if (in_array('status', $columns, true)) {
                    $row['status'] = 'active';
                }
                if (in_array('is_allowed', $columns, true)) {
                    $row['is_allowed'] = 1;
                }
                if (in_array('company_id', $columns, true) && $companyId !== null) {
                    $row['company_id'] = $companyId;
                }
                if (in_array('created_at', $columns, true)) {
                    $row['created_at'] = $now;
                }
                if (in_array('updated_at', $columns, true)) {
                    $row['updated_at'] = $now;
                }

                DB::table('module_settings')->insert($row);
                $inserted++;
            }
        }

        return [
            'title' => 'Seed module_settings',
            'detail' => $inserted . ' module_settings row(s) inserted for admin/employee/client visibility.',
            'mode' => 'safe',
        ];
    }

    private function seedCompanyModuleSettings(string $moduleName): ?array
    {
        if (!Schema::hasTable('company_module_settings') || !Schema::hasTable('companies')) {
            return null;
        }

        $columns = Schema::getColumnListing('company_module_settings');
        if (!in_array('company_id', $columns, true) || !in_array('module_name', $columns, true)) {
            return [
                'title' => 'Seed company_module_settings',
                'detail' => 'Skipped because company_module_settings schema is not compatible.',
                'mode' => 'safe',
            ];
        }

        $inserted = 0;
        $now = now();
        foreach (DB::table('companies')->pluck('id') as $companyId) {
            $exists = DB::table('company_module_settings')->where('company_id', $companyId)->where('module_name', $moduleName)->exists();
            if ($exists) {
                continue;
            }
            $row = [
                'company_id' => $companyId,
                'module_name' => $moduleName,
            ];
            if (in_array('status', $columns, true)) {
                $row['status'] = 'active';
            }
            if (in_array('created_at', $columns, true)) {
                $row['created_at'] = $now;
            }
            if (in_array('updated_at', $columns, true)) {
                $row['updated_at'] = $now;
            }
            DB::table('company_module_settings')->insert($row);
            $inserted++;
        }

        return [
            'title' => 'Seed company_module_settings',
            'detail' => $inserted . ' company_module_settings row(s) inserted for tenant access.',
            'mode' => 'safe',
        ];
    }
}
