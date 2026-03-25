<?php

namespace App\Services\TitanAI;

use Illuminate\Support\Facades\DB;
use Throwable;

class FallbackResolver
{
    /**
     * Execute AI with fallback order: on-device -> local/Ollama -> cloud.
     *
     * @return array{ok:bool, data?:mixed, error?:string, attempts?:array<int,array<string,mixed>>}
     */
    public function resolve(array $payload, array $context = []): array
    {
        $attempts = [];

        $tiers = [
            ['tier' => 'device', 'handler' => fn () => $this->callOnDevice($payload, $context)],
            ['tier' => 'local', 'handler' => fn () => $this->callLocalHost($payload, $context)],
            ['tier' => 'cloud', 'handler' => fn () => $this->callCloud($payload, $context)],
        ];

        foreach ($tiers as $tier) {
            $result = $this->attemptTier($tier['tier'], $tier['handler']);
            $attempts[] = $result['attempt'];

            if ($result['attempt']['status'] === 'ok') {
                $this->logRuntimeMeta('ai_fallback', [
                    'tier' => $tier['tier'],
                    'status' => 'ok',
                ]);

                return [
                    'ok' => true,
                    'data' => $result['attempt']['response'] ?? null,
                    'attempts' => $attempts,
                ];
            }
        }

        $this->logRuntimeMeta('ai_fallback', [
            'status' => 'failed',
            'attempts' => $attempts,
        ]);

        return [
            'ok' => false,
            'error' => 'All AI execution tiers failed. Check device AI availability, local host connectivity, and cloud service status.',
            'attempts' => $attempts,
        ];
    }

    /**
     * Stub: implement native/on-device call.
     */
    protected function callOnDevice(array $payload, array $context): array
    {
        return ['ok' => false, 'error' => 'on-device AI not implemented'];
    }

    /**
     * Stub: implement local/Ollama call.
     */
    protected function callLocalHost(array $payload, array $context): array
    {
        return ['ok' => false, 'error' => 'local AI host not implemented'];
    }

    /**
     * Stub: implement cloud call.
     */
    protected function callCloud(array $payload, array $context): array
    {
        return ['ok' => false, 'error' => 'cloud AI not implemented'];
    }

    /**
     * Attempt a tier and log the result to runtime metadata.
     */
    protected function attemptTier(string $tier, callable $callback): array
    {
        $attempt = [
            'tier' => $tier,
            'status' => 'failed',
            'response' => null,
            'error' => null,
        ];

        try {
            $response = $callback();
            if (!empty($response['ok'])) {
                $attempt['status'] = 'ok';
                $attempt['response'] = $response['data'] ?? $response ?? null;
            } else {
                $attempt['error'] = $response['error'] ?? 'unknown error';
            }
        } catch (Throwable $e) {
            $attempt['error'] = $e->getMessage();
        }

        $this->logRuntimeMeta('ai_attempt', $attempt);

        return ['attempt' => $attempt];
    }

    protected function logRuntimeMeta(string $metaKey, array $value): void
    {
        DB::table('tz_runtime_meta')->insert([
            'category' => 'titan_ai',
            'meta_key' => $metaKey,
            'meta_value' => $value,
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }
}
