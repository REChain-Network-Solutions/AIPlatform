<?php

namespace App\Extensions\AISocialMedia\System\Models;

use App\Extensions\AISocialMedia\System\Enums\Platform;
use App\Models\User;
use Illuminate\Database\Eloquent\Casts\Attribute;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Fluent;

class AutomationPlatform extends Model
{
    protected $fillable = [
        'user_id',
        'platform',
        'credentials',
        'connected_at',
        'expires_at',
    ];

    protected $casts = [
        'credentials' => 'json',
        'platform'    => Platform::class,
        'expires_at'  => 'datetime',
    ];

    protected $appends = [
        'fluent_credentials',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class, 'user_id');
    }

    public function fluentCredentials(): Attribute
    {
        return Attribute::make(
            get: fn () => new Fluent($this->credentials ?: []),
        );
    }

    public function getCredentialValue($key)
    {
        return $this->fluent_credentials?->get($key);
    }
}
