<?php

namespace Modules\Quotes\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Quote extends Model
{
    protected $table = 'quotes';

    protected $fillable = [
        'number','client_id','currency','status','valid_until','notes',
        'subtotal','tax_total','grand_total'
    ];

    protected $casts = [
        'valid_until' => 'date',
        'subtotal' => 'decimal:2',
        'tax_total' => 'decimal:2',
        'grand_total' => 'decimal:2',
    ];

    public function items(): HasMany
    {
        return $this->hasMany(QuoteItem::class, 'quote_id');
    }
}
