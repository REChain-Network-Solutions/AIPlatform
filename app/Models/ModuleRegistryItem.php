<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ModuleRegistryItem extends Model
{
    protected $table = 'module_registry_items';

    protected $fillable = [
        'feed_id',
        'module_name',
        'slug',
        'latest_version',
        'versions',
        'compatible_with',
        'dependencies',
        'distribution',
        'signature',
        'status',
        'last_seen_at',
        'meta',
    ];

    protected $casts = [
        'versions' => 'array',
        'compatible_with' => 'array',
        'dependencies' => 'array',
        'distribution' => 'array',
        'signature' => 'array',
        'last_seen_at' => 'datetime',
        'meta' => 'array',
    ];

    public function feed()
    {
        return $this->belongsTo(ModuleRegistryFeed::class, 'feed_id');
    }
}
