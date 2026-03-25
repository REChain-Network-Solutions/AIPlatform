<?php

declare(strict_types=1);

namespace App\Services\Ai;

use Illuminate\Support\Facades\Log;

class DeviceCompletionService
{
    public function enabled(): bool
    {
        return (bool) config('ai.routing.device_enabled', false);
    }

    public function complete(string $systemPrompt, string $userContent): ?string
    {
        if (! $this->enabled()) {
            return null;
        }

        // Hook for future on-device inference.
        Log::info('DeviceCompletionService invoked but no device runtime is configured.');

        return null;
    }
}
