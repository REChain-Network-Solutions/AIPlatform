<?php

namespace Modules\EInvoice\Entities;

use Illuminate\Database\Eloquent\Model;

class EInvoiceNote extends Model
{
    protected $table = 'einvoice_ai_notes';

    protected $fillable = [
        'invoice_id',
        'user_id',
        'content',
    ];
}
