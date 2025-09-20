<?php

namespace App\Services\PaymentGateways;

use App\Models\Gateways;
use App\Services\PaymentGateways\Contracts\CreditUpdater;

class TwoCheckoutService
{
    use CreditUpdater;

    protected static string $GATEWAY_CODE = 'twocheckout';

    protected static string $GATEWAY_NAME = '2 Checkout';

    private static ?Gateways $gateway = null;
}
