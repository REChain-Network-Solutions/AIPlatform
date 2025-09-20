<?php

namespace App\Packages\FalAI;

use App\Domains\Entity\Enums\EntityEnum;
use App\Packages\FalAI\API\BaseApiClient;
use App\Packages\FalAI\Contracts\TextToVideoModelInterface;
use App\Packages\FalAI\Models\FastVeo3;
use App\Packages\FalAI\Models\Kling;
use App\Packages\FalAI\Models\Veed;
use App\Packages\FalAI\Models\Veo3;

class FalAIService
{
    protected BaseApiClient $client;

    public function __construct(
        string $falKey,
        string $apiBaseUrl = 'https://queue.fal.run'
    ) {
        set_time_limit(0);
        ini_set('memory_limit', '-1');
        ini_set('max_execution_time', 3600);

        $this->client = new BaseApiClient($falKey, $apiBaseUrl);
    }

    /** text to video model */
    public function textToVideoModel(EntityEnum $model): TextToVideoModelInterface
    {
        return match ($model) {
            EntityEnum::KLING_VIDEO => new Kling($this->client),
            EntityEnum::VEED        => new Veed($this->client),
            EntityEnum::VEO_3       => new Veo3($this->client),
            EntityEnum::VEO_3_FAST  => new FastVeo3($this->client)
        };
    }
}
