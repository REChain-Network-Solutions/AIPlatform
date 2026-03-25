<?php

namespace Modules\Quotes\Entities;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class PriceListItem extends Model
{
    protected $table = 'quote_price_list_items';
    protected $fillable = ['price_list_id','item_name','unit_price','tax_rate'];

    public function priceList(): BelongsTo
    {
        return $this->belongsTo(PriceList::class, 'price_list_id');
    }
}
