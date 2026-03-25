<?php

namespace Workdo\Pos\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class PosProduct extends Model
{
    use HasFactory;

    protected $fillable = [
        'product_type',
        'product_id',
        'pos_id',
        'quantity',
        'tax',
        'discount',
        'total',
        'workspace',
    ];

    public function product(){
        return $this->hasOne(\Workdo\ProductService\Entities\ProductService::class, 'id', 'product_id');
    }

}
