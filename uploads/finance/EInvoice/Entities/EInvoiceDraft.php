<?php

namespace Modules\EInvoice\Entities;

use Illuminate\Database\Eloquent\Model;

class EInvoiceDraft extends Model
{
    protected $table = 'einvoice_ai_drafts';

    protected $fillable = [
        'user_id',
        'client_id',
        'payload',
    ];

    protected $casts = [
        'payload' => 'array',
    ];
}
