<?php

namespace Modules\EInvoice\Entities;

use App\Models\BaseModel;
use App\Traits\HasCompany;

class EInvoiceCompanySetting extends BaseModel
{
    use HasCompany;

    protected $guarded = ['id'];
}
