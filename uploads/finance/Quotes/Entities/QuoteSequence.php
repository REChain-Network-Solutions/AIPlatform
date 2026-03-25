<?php

namespace Modules\Quotes\Entities;

use Illuminate\Database\Eloquent\Model;

class QuoteSequence extends Model
{
    protected $table = 'quote_sequences';
    protected $fillable = ['series','year','next_number'];
}
