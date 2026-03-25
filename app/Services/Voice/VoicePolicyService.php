<?php

declare(strict_types=1);

namespace App\Services\Voice;

use Illuminate\Http\Request;
use Symfony\Component\HttpKernel\Exception\HttpException;

class VoicePolicyService
{
    public function requiresConsent(): bool
    {
        return (bool) config('voice.require_consent', true);
    }

    public function allowRemoteStreaming(): bool
    {
        return (bool) config('voice.allow_remote_stream', true);
    }

    public function assertConsent(Request $request): void
    {
        if ($this->requiresConsent() && ! $request->boolean('consent')) {
            throw new HttpException(403, __('Voice streaming requires user consent.'));
        }
    }
}
