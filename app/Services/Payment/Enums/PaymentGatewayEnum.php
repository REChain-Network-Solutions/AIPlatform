<?php

namespace App\Services\Payment\Enums;

use App\Enums\Contracts\WithStringBackedEnum;
use App\Enums\Traits\EnumTo;
use App\Enums\Traits\StringBackedEnumTrait;

enum PaymentGatewayEnum: string implements WithStringBackedEnum
{
    use EnumTo;
    use StringBackedEnumTrait;

    // the order of these cases matters, as they are used in the UI
    case Paypal = 'paypal';
    case Stripe = 'stripe';
    case TwoCheckout = '2checkout';
    case YoKassa = 'yokassa';
    case Iyzico = 'iyzico';
    case Paystack = 'paystack';
    case BankTransfer = 'banktransfer';
    case RevenueCat = 'revenuecat';
    case CoinGate = 'coingate';
    case Paddle = 'paddle';
    case Razorpay = 'razorpay';
    case CryptoMus = 'cryptomus';
    case Midtrans = 'midtrans';

    public function label(): string
    {
        return match ($this) {
            self::TwoCheckout  => __('2Checkout'),
            self::Stripe       => __('Stripe'),
            self::Paypal       => __('PayPal'),
            self::YoKassa      => __('YooKassa'),
            self::Iyzico       => __('Iyzico'),
            self::Paystack     => __('Paystack'),
            self::BankTransfer => __('Bank Transfer'),
            self::RevenueCat   => __('RevenueCat'),
            self::CoinGate     => __('CoinGate'),
            self::Paddle       => __('Paddle'),
            self::Razorpay     => __('Razorpay'),
            self::CryptoMus    => __('CryptoMus'),
            self::Midtrans     => __('Midtrans'),
        };
    }

    public static function isRefactored($gatewayCode): bool
    {
        return $gatewayCode === self::TwoCheckout->value;
    }

    public static function activeGateways(): array
    {
        $disabledGateways = [
            // Add a gateway to this list to disable it
            // self::Paypal, // Example: Disable PayPal

            // add the extension check here to extract gatway if extension not installed
            ! class_exists(\App\Services\PaymentGateways\MidtransService::class) ? self::Midtrans : null,
            ! class_exists(\App\Extensions\Cryptomus\System\Services\CryptomusService::class) ? self::CryptoMus : null,
        ];

        return array_values(array_map(
            static fn ($case) => $case->value,
            array_filter(self::cases(), static fn ($case) => ! in_array($case, $disabledGateways, true))
        ));
    }

    public function gatewayDefinition(): ?array
    {
        $base = [
            'code'                  => $this->value,
            'title'                 => $this->label(),
            'available'             => 1,
            'mode'                  => 1,
            'sandbox_client_id'     => 1,
            'sandbox_client_secret' => 1,
            'tax'                   => 1,
            'automate_tax'          => 0,
            'live_client_id'        => 1,
            'live_client_secret'    => 1,
            'currency'              => 1,
            'whiteLogo'             => 0,
            'live_app_id'           => 0,
            'sandbox_app_id'        => 0,
            'active'                => 0,
            'currency_locale'       => 0,
            'base_url'              => 0,
            'sandbox_url'           => 0,
            'locale'                => 0,
            'validate_ssl'          => 0,
            'logger'                => 0,
            'notify_url'            => 0,
            'webhook_secret'        => 0,
            'bank_account_details'  => 0,
            'bank_account_other'    => 0,
            'link'                  => '',
            'img'                   => '',
        ];

        return match ($this) {
            self::TwoCheckout => array_merge($base, [
                'link'                  => 'https://2checkout.com/',
                'img'                   => '/assets/img/payments/2checkout.svg',
                'live_app_id'           => 1,
                'mode'                  => 0,
                'live_client_id'        => 0,
                'sandbox_client_id'     => 0,
                'sandbox_client_secret' => 0,
            ]),
            self::Stripe => array_merge($base, [
                'link'         => 'https://stripe.com/',
                'img'          => '/assets/img/payments/stripe.svg',
                'base_url'     => 1,
                'automate_tax' => 1,
            ]),
            self::Paypal => array_merge($base, [
                'link'                => 'https://www.paypal.com/',
                'img'                 => '/assets/img/payments/paypal.svg',
                'currency_locale'     => 1,
                'live_app_id'         => 1,
                'base_url'            => 0,
            ]),
            self::YoKassa => array_merge($base, [
                'link' => 'https://yokassa.ru/',
                'img'  => '/assets/img/payments/yokassa.svg',
            ]),
            self::Iyzico => array_merge($base, [
                'link'        => 'https://www.iyzico.com/',
                'img'         => '/assets/img/payments/iyzico.svg',
                'base_url'    => 1,
                'sandbox_url' => 1,
            ]),
            self::Paystack => array_merge($base, [
                'link' => 'https://paystack.com/',
                'img'  => '/assets/img/payments/paystack-2.svg',
            ]),
            self::BankTransfer => array_merge($base, [
                'link'                  => '',
                'img'                   => '/assets/img/payments/banktransfer.png',
                'mode'                  => 0,
                'sandbox_client_id'     => 0,
                'sandbox_client_secret' => 0,
                'live_client_id'        => 0,
                'live_client_secret'    => 0,
                'currency_locale'       => 0,
                'bank_account_details'  => 1,
                'bank_account_other'    => 1,
            ]),
            self::RevenueCat => array_merge($base, [
                'link'               => 'https://www.revenuecat.com/',
                'img'                => '/assets/img/payments/revenuecat.png',
                'active'             => 1,
                'mode'               => 0,
                'live_client_secret' => 0,
                'live_client_id'     => 1,
                'tax'                => 1,
                'currency'           => 0,
            ]),
            self::CoinGate => array_merge($base, [
                'link'                  => 'https://coingate.com/',
                'img'                   => '/assets/img/payments/coingate.svg',
                'currency'              => 0,
                'sandbox_client_id'     => 0,
                'live_client_id'        => 0,
            ]),
            self::Paddle => array_merge($base, [
                'link' => 'https://paddle.com/',
                'img'  => custom_theme_url('/assets/img/payments/paddle.svg'),
            ]),
            self::Razorpay => array_merge($base, [
                'link'     => 'https://razorpay.com/',
                'img'      => '/assets/img/payments/razorpay.svg',
                'currency' => 0,
            ]),
            // extensions to be added below
            self::Midtrans => class_exists(\App\Services\PaymentGateways\MidtransService::class)
                ? array_merge($base, \App\Services\PaymentGateways\MidtransService::gatewayDefinitionArray())
                : null,
            self::CryptoMus => class_exists(\App\Extensions\Cryptomus\System\Services\CryptomusService::class)
                ? array_merge($base, \App\Extensions\Cryptomus\System\Services\CryptomusService::gatewayDefinitionArray())
                : null,
        };
    }
}
