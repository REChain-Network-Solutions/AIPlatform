<?php

declare(strict_types=1);

namespace App\Models;

use App\Models\Concerns\HasCache;
use Illuminate\Database\Eloquent\Model;

class Gateways extends Model
{
    use HasCache;

    public static int $cacheTtl = 3600 * 24;

    public static string $cacheKey = 'cache_gateways_result';

    protected $guarded = [];

    protected $casts = [
        'is_active' => 'boolean',
    ];

    protected $fillable = [
        'code',
        'title',
        'is_active',
        'mode',
        'sandbox_client_id',
        'sandbox_client_secret',
        'sandbox_app_id',
        'live_client_id',
        'live_client_secret',
        'live_app_id',
        'payment_action',
        'currency',
        'currency_local',
        'notify_url',
        'base_url',
        'sandbox_url',
        'locale',
        'validate_ssl',
        'webhook_secret',
        'logger',
        'webhook_id',
        'tax',
        'automate_tax',
        'bank_account_details',
        'bank_account_other',
        'country_tax_enabled',
    ];

    protected static function booted(): void
    {
        parent::boot();

        static::saved(static fn () => static::forgetCache());
        static::deleted(static fn () => static::forgetCache());
    }

    public function isSandbox(): bool
    {
        return $this->mode === 'sandbox';
    }
}
