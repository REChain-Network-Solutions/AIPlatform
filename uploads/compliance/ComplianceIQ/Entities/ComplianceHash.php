<?php

namespace Modules\ComplianceIQ\Entities;

use Illuminate\Database\Eloquent\Model;

class ComplianceHash extends Model
{
    public $timestamps = false;
    protected $fillable = ['hashable_type','hashable_id','sha256','computed_at','status'];
    protected $casts = ['computed_at' => 'datetime'];
}
