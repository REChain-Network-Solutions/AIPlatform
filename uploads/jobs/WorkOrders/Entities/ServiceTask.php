<?php

namespace Modules\WorkOrders\Entities;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class ServiceTask extends Model
{
    use HasFactory;
    protected $fillable=[
        'service_id',
        'task',
        'duration',
        'description',
    ];
}
