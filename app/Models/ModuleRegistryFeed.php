<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleRegistryFeed extends Model
{
    protected $table = 'module_registry_feeds';

    protected $fillable = [
        'name',
        'slug',
        'source_type',
        'source_url',
        'auth_config',
        'is_active',
        'priority',
        'last_synced_at',
        'last_error',
        'sync_meta',
    ];

    protected $casts = [
        'auth_config' => 'array',
        'is_active' => 'boolean',
        'priority' => 'integer',
        'last_synced_at' => 'datetime',
        'sync_meta' => 'array',
    ];
}
