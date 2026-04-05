<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleMenuBlocker extends Model
{
    protected $guarded = [];

    protected $casts = [
        'details_json' => 'array',
        'is_auto_fixable' => 'boolean',
    ];
}
