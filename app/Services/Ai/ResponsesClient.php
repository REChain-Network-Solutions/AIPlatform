<?php

declare(strict_types=1);

namespace App\Services\Ai;

use Closure;
use Illuminate\Support\Facades\Log;
use OpenAI\Contracts\Resources\ResponsesContract;
use OpenAI\Laravel\Facades\OpenAI;
use OpenAI\Responses\Responses\CreateResponse;
use OpenAI\Responses\Responses\RetrieveResponse;
use RuntimeException;
use Throwable;

final class ResponsesClient
{
    public function __construct(private readonly ?ResponsesContract $responses = null) {}

    public function create(array $parameters): CreateResponse
    {
        return $this->guard(
            fn (): CreateResponse => $this->resource()->create($parameters),
            ['model' => $parameters['model'] ?? null]
        );
    }

    public function retrieve(string $responseId): RetrieveResponse
    {
        return $this->guard(
            fn (): RetrieveResponse => $this->resource()->retrieve($responseId),
            ['response_id' => $responseId]
        );
    }

    public function cancel(string $responseId): RetrieveResponse
    {
        return $this->guard(
            fn (): RetrieveResponse => $this->resource()->cancel($responseId),
            ['response_id' => $responseId]
        );
    }

    /**
     * Run a provider operation, logging technical failures while only
     * surfacing a displayable message to callers.
     *
     * @template T
     * @param  Closure(): T  $operation
     *
     * @return T
     */
    public function guard(Closure $operation, array $context = []): mixed
    {
        try {
            return $operation();
        } catch (Throwable $e) {
            Log::error('OpenAI Responses request failed', $context + [
                'error' => $e::class . ': ' . $e->getMessage(),
            ]);

            if ($e instanceof RuntimeException) {
                throw $e;
            }

            throw new RuntimeException(__('The AI provider returned an unexpected response. Please try again.'), previous: $e);
        }
    }

    private function resource(): ResponsesContract
    {
        return $this->responses ?? OpenAI::responses();
    }
}
