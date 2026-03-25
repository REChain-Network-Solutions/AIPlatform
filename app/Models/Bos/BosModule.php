<?php

namespace App\Models\Bos;

use Illuminate\Database\Eloquent\Model;

class BosModule extends Model
{
    protected $table = 'bos_modules';

    protected $fillable = [
        'key',
        'name',
        'description',
        'icon',
        'color',
        'route_prefix',
        'is_active',
        'is_core',
        'order',
        'settings',
    ];

    protected $casts = [
        'is_active' => 'boolean',
        'is_core'   => 'boolean',
        'settings'  => 'array',
    ];

    public static function activeModules()
    {
        return static::where('is_active', true)->orderBy('order')->get();
    }
}
