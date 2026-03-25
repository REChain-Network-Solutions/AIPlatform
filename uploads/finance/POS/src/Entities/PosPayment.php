<?php

namespace Workdo\Pos\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class PosPayment extends Model
{
    use HasFactory;

    protected $fillable = [
        'pos_id',
        'date',
        'amount',
        'discount',
        'discount_amount',
        'created_by',
    ];

}
