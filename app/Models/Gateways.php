<?php

declare(strict_types=1);

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Gateways extends Model
{
    protected $guarded = [];

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

    public function isSandbox(): bool
    {
        return $this->mode === 'sandbox';
    }
}
