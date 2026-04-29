<?php

use App\Extensions\Cryptomus\System\Services\CryptomusService;
use App\Services\PaymentGateways\CoinbaseService;
use App\Services\PaymentGateways\CoingateService;
use App\Services\PaymentGateways\FreeService;
use App\Services\PaymentGateways\IyzicoService;
use App\Services\PaymentGateways\MidtransService;
use App\Services\PaymentGateways\PaddleService;
use App\Services\PaymentGateways\PayPalService;
use App\Services\PaymentGateways\PaystackService;
use App\Services\PaymentGateways\RazorpayService;
use App\Services\PaymentGateways\RevenueCatService;
use App\Services\PaymentGateways\StripeService;
use App\Services\PaymentGateways\TransferService;
use App\Services\PaymentGateways\YokassaService;

return [

    /*
    |--------------------------------------------------------------------------
    | Third Party Services
    |--------------------------------------------------------------------------
    |
    | This file is for storing the credentials for third party services such
    | as Mailgun, Postmark, AWS and more. This file provides the de facto
    | location for this type of information, allowing packages to have
    | a conventional file to locate the various service credentials.
    |
    */

    'mailgun' => [
        'domain'   => env('MAILGUN_DOMAIN'),
        'secret'   => env('MAILGUN_SECRET'),
        'endpoint' => env('MAILGUN_ENDPOINT', 'api.mailgun.net'),
        'scheme'   => 'https',
    ],

    'postmark' => [
        'token' => env('POSTMARK_TOKEN'),
    ],

    'ses' => [
        'key'    => env('AWS_ACCESS_KEY_ID'),
        'secret' => env('AWS_SECRET_ACCESS_KEY'),
        'region' => env('AWS_DEFAULT_REGION', 'us-east-1'),
    ],
    'github' => [
        'client_id'     => env('GITHUB_CLIENT_ID'),
        'client_secret' => env('GITHUB_CLIENT_SECRET'),
        'redirect'      => '/github/callback',
    ],
    'google' => [
        'client_id'     => env('GOOGLE_CLIENT_ID'),
        'client_secret' => env('GOOGLE_CLIENT_SECRET'),
        'redirect'      => '/google/callback',
    ],
    'facebook' => [
        'client_id'     => env('FACEBOOK_CLIENT_ID'),
        'client_secret' => env('FACEBOOK_CLIENT_SECRET'),
        'redirect'      => '/facebook/callback',
    ],
    'apple' => [
        'client_id' => env('APPLE_BUNDLE_ID'),
    ],
    'recaptcha' => [
        'key'    => env('RECAPTCHA_SITE_KEY'),
        'secret' => env('RECAPTCHA_SECRET_KEY'),
    ],

    /*
    |--------------------------------------------------------------------------
    | Payment Gateways Services
    |--------------------------------------------------------------------------
    */

    'stripe' => [
        'class' => StripeService::class,
    ],

    'paypal' => [
        'class' => PayPalService::class,
    ],

    'paystack' => [
        'class' => PaystackService::class,
    ],

    'yokassa' => [
        'class' => YokassaService::class,
    ],

    'iyzico' => [
        'class' => IyzicoService::class,
    ],

    'razorpay' => [
        'class' => RazorpayService::class,
    ],

    'banktransfer' => [
        'class' => TransferService::class,
    ],

    'freeservice' => [
        'class' => FreeService::class,
    ],

    'revenuecat' => [
        'class' => RevenueCatService::class,
    ],

    'coinbase' => [
        'class' => CoinbaseService::class,
    ],

    'coingate' => [
        'class' => CoingateService::class,
    ],

    'paddle' => [
        'class' => PaddleService::class,
    ],

    'cryptomus' => [
        'class' => CryptomusService::class,
    ],

    'midtrans' => [
        'class' => MidtransService::class,
    ],
];
