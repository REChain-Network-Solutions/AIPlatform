<?php

declare(strict_types=1);

namespace App\Services\Ai;

use App\Helpers\Classes\ApiHelper;
use App\Models\User;

class ApiKeyResolver
{
    public function applyOpenAiKey(?User $user = null): ?string
    {
        // ApiHelper already respects user_api_option in settings.
        return ApiHelper::setOpenAiKey();
    }

    public function applyAnthropicKey(?User $user = null): ?string
    {
        return ApiHelper::setAnthropicKey();
    }

    public function applyGeminiKey(?User $user = null): ?string
    {
        return ApiHelper::setGeminiKey();
    }

    public function applyDeepseekKey(?User $user = null): ?string
    {
        return ApiHelper::setDeepseekKey();
    }

    public function applyXAiKey(?User $user = null): ?string
    {
        return ApiHelper::setXAiKey();
    }
}
