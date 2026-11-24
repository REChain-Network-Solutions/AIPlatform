<?php

namespace App\Http\Controllers\OpenAi;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Helpers\Classes\ApiHelper;
use App\Http\Controllers\Controller;
use Exception;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class RealtimeClientSecretController extends Controller
{
    /**
     * Handle the incoming request.
     */
    public function __invoke(): JsonResponse
    {
        try {
            Entity::driver(EntityEnum::GPT_REALTIME)->redirectIfNoCreditBalance();
        } catch (Exception $exception) {
            return response()->json(['message' => $exception->getMessage()], 402);
        }

        $response = Http::withToken(ApiHelper::setOpenAiKey())
            ->acceptJson()
            ->connectTimeout(5)
            ->timeout(15)
            ->post('https://api.openai.com/v1/realtime/client_secrets', [
                'session' => [
                    'type'  => 'realtime',
                    'model' => 'gpt-realtime',
                ],
            ]);

        if ($response->failed()) {
            Log::warning('OpenAI Realtime client-secret request failed.', [
                'status' => $response->status(),
            ]);

            return response()->json(['message' => __('Unable to start a realtime session.')], 502);
        }

        $secret = $response->json('value') ?? $response->json('client_secret.value');
        $expiresAt = $response->json('expires_at') ?? $response->json('client_secret.expires_at');

        if (! is_string($secret) || $secret === '') {
            return response()->json(['message' => __('The realtime provider returned an invalid response.')], 502);
        }

        return response()->json([
            'client_secret' => $secret,
            'expires_at'    => $expiresAt,
        ]);
    }
}
