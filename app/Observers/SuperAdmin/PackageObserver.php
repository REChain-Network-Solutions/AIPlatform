<?php

namespace App\Observers\SuperAdmin;

use App\Models\PackageUpdateNotify;
use App\Models\SuperAdmin\Package;
use App\Observers\CompanyObserver;

class PackageObserver
{

    public function saving(Package $package)
    {
        if (($package->is_free || $package->default === 'yes') && $package->package != 'lifetime') {
            $package->monthly_status = 1;
            $package->annual_status = 1;
        }
    }

    public function updated(Package $package)
    {
        $package->companies->each(function ($company) use ($package) {
            if ($package->isDirty('module_in_package')) {
                (new CompanyObserver())->updateModuleSettings($company);
            }

            $companyEmployeesCount = $company->employees()->count();

            if ($companyEmployeesCount <= $package->max_employees) {
                PackageUpdateNotify::where('company_id', $company->id)->delete();
            }

            clearCompanyValidPackageCache($company->id);
        });

    }

}
