<?php

namespace App\Models;

use App\Scopes\CompanyScope;
use App\Traits\HasCompany;

/**
 * App\Models\ModuleSetting
 *
 * @property int $id
 * @property string $module_name
 * @property string $status
 * @property string $type
 * @property \Illuminate\Support\Carbon|null $created_at
 * @property \Illuminate\Support\Carbon|null $updated_at
 * @property-read mixed $icon
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting newModelQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting newQuery()
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting query()
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereCreatedAt($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereId($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereModuleName($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereStatus($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereType($value)
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereUpdatedAt($value)
 * @property int|null $company_id
 * @property-read \App\Models\Company|null $company
 * @method static \Illuminate\Database\Eloquent\Builder|ModuleSetting whereCompanyId($value)
 * @mixin \Eloquent
 */
class ModuleSetting extends BaseModel
{

    use HasCompany;

    const CLIENT_MODULES = [
        'projects',
        'tickets',
        'invoices',
        'estimates',
        'events',
        'messages',
        'tasks',
        'timelogs',
        'contracts',
        'notices',
        'payments',
        'orders',
        'knowledgebase',
    ];

    const OTHER_MODULES = [
        'clients',
        'employees',
        'attendance',
        'expenses',
        'leaves',
        'leads',
        'holidays',
        'products',
        'reports',
        'settings',
        'bankaccount'
    ];

    protected $guarded = ['id'];

    public static function checkModule($moduleName)
    {

        $module = ModuleSetting::where('module_name', $moduleName);

        if (in_array('admin', user_roles())) {
            $module = $module->where('type', 'admin');
        } elseif (in_array('client', user_roles())) {
            $module = $module->where('type', 'client');
        } elseif (in_array('employee', user_roles())) {
            $module = $module->where('type', 'employee');
        }

        $module = $module->where('status', 'active');

        $module = $module->first();

        return (bool)$module;
    }

    public static function addCompanyIdToNullModule($company, $module)
    {
        // This is done for existing module settings. This will update the company id with 1
        // for existing module rather creating new module setting
        if ($company->id == 1) {
            ModuleSetting::withoutGlobalScope(CompanyScope::class)->where('module_name', $module)
                ->whereNull('company_id')
                ->update(['company_id' => $company->id]);
        }
    }

    public static function createRoleSettingEntry($module, $roles, $company)
    {
        self::addCompanyIdToNullModule($company, $module);

        // WORKSUITESAAS
        $moduleInPackage = collect(json_decode($company->package->module_in_package));

        foreach ($roles as $role) {
            $data = ModuleSetting::withoutGlobalScope(CompanyScope::class)
                ->where('module_name', $module)
                ->where('type', $role)
                ->where('company_id', $company->id)
                ->first();

            if (!$data) {

                ModuleSetting::create([
                    'module_name' => $module,
                    'type' => $role,
                    'company_id' => $company->id,
                    'status' => 'active',
                    'is_allowed' => $moduleInPackage->contains($module) ? 1 : 0,
                ]);
            }
        }

        PermissionRole::insertModuleRolePermission($module, $company->id);
    }
}
