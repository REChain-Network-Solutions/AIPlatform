<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleVisibilityAudit extends Model
{
    protected $guarded = [];

    protected $casts = [
        'payload' => 'array',
    ];
}
