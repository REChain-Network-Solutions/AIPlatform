<?php

namespace App\Services\Payment\Factories;

use App\Services\Payment\Contracts\PaymentGatewayInterface;
use App\Services\Payment\Enums\PaymentGatewayEnum;
use App\Services\Payment\Gateways\TwoCheckoutGateway;
use InvalidArgumentException;

class GatewayFactory
{
    public static function make(PaymentGatewayEnum $gateway): PaymentGatewayInterface
    {
        return match ($gateway) {
            PaymentGatewayEnum::TwoCheckout => new TwoCheckoutGateway,
            // Add other gateways here as needed
            default => throw new InvalidArgumentException("Unsupported payment gateway: {$gateway->value}"),
        };
    }
}
