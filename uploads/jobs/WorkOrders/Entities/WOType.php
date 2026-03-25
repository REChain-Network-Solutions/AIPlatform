<?php

namespace Modules\WorkOrders\Entities;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class WOType extends Model
{
    use HasFactory;
    protected $fillable=[
        'type',
        'parent_id',
    ];
}
