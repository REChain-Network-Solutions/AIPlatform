<?php

declare(strict_types=1);

namespace App\Models;

use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Model;

class SharedCreditCost extends Model
{
    protected $fillable = [
        'entity_key',
        'engine_key',
        'feature_type',
        'base_cost',
        'quality_high_multiplier',
        'quality_low_multiplier',
        'is_active',
    ];

    protected $casts = [
        'base_cost'               => 'float',
        'quality_high_multiplier' => 'float',
        'quality_low_multiplier'  => 'float',
        'is_active'               => 'boolean',
    ];

    public function scopeActive(Builder $query): Builder
    {
        return $query->where('is_active', true);
    }
}
