<?php

namespace Modules\EInvoice\Entities;

use App\Models\BaseModel;
use App\Scopes\CompanyScope;

class EInvoiceSetting extends BaseModel
{

    protected $guarded = ['id'];

    const MODULE_NAME = 'einvoice';

    public static function addModuleSetting($company)
    {
        $setting = EInvoiceCompanySetting::withoutGlobalScope(CompanyScope::class)->where('company_id', $company->id)->first();

        if (! $setting) {
            $newSetting = new EInvoiceCompanySetting;
            $newSetting->company_id = $company->id;
            $newSetting->saveQuietly();
        }
    }

}

