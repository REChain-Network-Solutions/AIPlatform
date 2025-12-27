<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Coupon extends Model
{
    use HasFactory;

    protected $fillable = [
        'name',
        'code',
        'discount',
        'is_offer',
        'is_offer_fixed_price',
        'limit',
        'created_by',
        'offer_id',
    ];

    public function createdBy()
    {
        return $this->belongsTo(User::class, 'created_by');
    }

    public function usersUsed()
    {
        return $this->belongsToMany(User::class, 'coupon_users')
            ->withTimestamps();
    }
}
