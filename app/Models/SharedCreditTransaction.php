<?php

declare(strict_types=1);

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class SharedCreditTransaction extends Model
{
    protected $fillable = [
        'user_id',
        'entity_key',
        'engine_key',
        'feature_type',
        'action_type',
        'amount',
        'balance_after',
        'unit_cost',
        'quantity',
        'quality',
        'metadata',
        'description',
    ];

    protected $casts = [
        'amount'        => 'float',
        'balance_after' => 'float',
        'unit_cost'     => 'float',
        'quantity'      => 'float',
        'metadata'      => 'array',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
