<?php

namespace Modules\ComplianceIQ\Entities;

use Illuminate\Database\Eloquent\Model;

class ComplianceReport extends Model
{
    protected $fillable = [
        'title','period_start','period_end','status',
        'signed_off_by','signed_off_at','filters','summary'
    ];
    protected $casts = [
        'filters' => 'array',
        'summary' => 'array',
        'period_start' => 'date',
        'period_end' => 'date',
        'signed_off_at' => 'datetime',
    ];

    public function annotations()
    {
        return $this->hasMany(ComplianceAnnotation::class, 'report_id');
    }
}
