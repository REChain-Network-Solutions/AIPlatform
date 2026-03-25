<?php

namespace Modules\WorkOrders\Entities;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class WOServicePart extends Model
{
    use HasFactory;
    protected $fillable=[
        'wo_id',
        'service_part_id',
        'quantity',
        'amount',
        'type',
        'description',
    ];

    public function serviceParts()
    {
        return $this->hasOne('Modules\WorkOrders\Entities\ServicePart','id','service_part_id');
    }

    public function serviceTasks()
    {
        return $this->hasMany('Modules\WorkOrders\Entities\ServiceTask','service_id','id');
    }
}
