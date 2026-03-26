<?php

namespace Modules\ComplianceIQ\Entities;

use Illuminate\Database\Eloquent\Model;

class ComplianceAnnotation extends Model
{
    protected $fillable = ['report_id','user_id','note'];
}
