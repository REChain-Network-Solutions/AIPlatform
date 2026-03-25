<?php

namespace App\Services\TitanAI;

use App\Services\TitanRuntime\TelemetryService;
use Illuminate\Support\Facades\DB;
use Throwable;

class FallbackResolver
{
    private const NANOS_PER_MILLI = 1_000_000;

    public function __construct(private ?TelemetryService $telemetry = null)
    {
        $this->telemetry = $telemetry ?? new TelemetryService();
    }

    /**
     * Execute AI with fallback order: on-device -> local/Ollama -> cloud.
     *
     * @return array{ok:bool, data?:mixed, error?:string, attempts?:array<int,array<string,mixed>>}
     */
    public function resolve(string $prompt, array $tools = [], array $policy = []): array
    {
        $attempts = [];

        $tiers = [
            ['tier' => 'device', 'handler' => fn () => $this->callOnDevice($prompt, $tools, $policy)],
            ['tier' => 'local', 'handler' => fn () => $this->callLocalHost($prompt, $tools, $policy)],
            ['tier' => 'cloud', 'handler' => fn () => $this->callCloud($prompt, $tools, $policy)],
        ];

        foreach ($tiers as $tier) {
            $result = $this->attemptTier($tier['tier'], $tier['handler']);
            $attempts[] = $result['attempt'];
            $latency = $result['attempt']['latency_ms'] ?? null;

            if ($result['attempt']['status'] === 'ok') {
                $this->logRuntimeMeta('ai_fallback', [
                    'tier' => $tier['tier'],
                    'status' => 'ok',
                    'latency_ms' => $latency,
                ]);

                $this->telemetry->logProviderSelection(
                    subsystem: 'titan_ai',
                    executionLayer: $tier['tier'],
                    success: true,
                    metadata: [
                        'provider' => $tier['tier'],
                        'latency_ms' => $latency,
                    ]
                );

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
    protected function callOnDevice(string $prompt, array $tools, array $policy): array
    {
        return ['ok' => false, 'error' => 'on-device AI not implemented'];
    }

    /**
     * Stub: implement local/Ollama call.
     */
    protected function callLocalHost(string $prompt, array $tools, array $policy): array
    {
        return ['ok' => false, 'error' => 'local AI host not implemented'];
    }

    /**
     * Stub: implement cloud call.
     */
    protected function callCloud(string $prompt, array $tools, array $policy): array
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
            'latency_ms' => null,
        ];

        try {
            $start = hrtime(true);
            $response = $callback();
            $attempt['latency_ms'] = (int) ((hrtime(true) - $start) / self::NANOS_PER_MILLI);

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
        $this->telemetry->logExecution(
            subsystem: 'titan_ai',
            event: 'attempt',
            executionLayer: $tier,
            success: $attempt['status'] === 'ok',
            latencyMs: $attempt['latency_ms'],
            metadata: $attempt
        );

        return ['attempt' => $attempt];
    }

    protected function logRuntimeMeta(string $metaKey, array $value): void
    {
        DB::table('tz_runtime_meta')->insert([
            'subsystem' => 'titan_ai',
            'event' => $metaKey,
            'execution_layer' => $value['tier'] ?? null,
            'latency_ms' => $value['latency_ms'] ?? null,
            'success_flag' => ($value['status'] ?? null) === 'ok',
            'metadata_json' => json_encode($value),
            'created_at' => now(),
            'updated_at' => now(),
        ]);
    }
}
