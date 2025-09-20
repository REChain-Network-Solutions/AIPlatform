<?php

namespace App\Jobs;

use App\Models\User;
use App\Services\Payment\Enums\PaymentGatewayEnum;
use App\Services\Payment\Factories\GatewayFactory;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

class ProcessGatewayCustomerJob implements ShouldQueue
{
    use Dispatchable;
    use InteractsWithQueue;
    use Queueable;
    use SerializesModels;

    protected PaymentGatewayEnum $gatewayEnum;

    protected User $user;

    public function __construct(PaymentGatewayEnum $gatewayEnum, User $user)
    {
        $this->gatewayEnum = $gatewayEnum;
        $this->user = $user;
    }

    public function handle(): void
    {
        $gateway = GatewayFactory::make($this->gatewayEnum);
        $gateway->createCustomer($this->user);
    }
}
