<?php

namespace App\Services\Payment\Contracts;

use App\Models\Gateways;

abstract class AbstractPaymentGateway
{
    protected static array $gatewayCache = [];

    public function __construct()
    {
        set_time_limit(0);
        ini_set('max_execution_time', 9000);

        $this->loadConfig();
    }

    /**
     * Use this method to load configuration settings
     * from the config files or settings table.
     */
    abstract protected function loadConfig(): void;

    public static function getGateway(): ?Gateways
    {
        $code = static::enum()->value;

        if (! isset(self::$gatewayCache[$code])) {
            self::$gatewayCache[$code] = Gateways::where('code', $code)->first();
        }

        return self::$gatewayCache[$code];
    }
}
