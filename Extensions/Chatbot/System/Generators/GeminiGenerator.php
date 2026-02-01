<?php

namespace App\Extensions\Chatbot\System\Generators;

use App\Domains\Engine\Services\GeminiService;
use App\Extensions\Chatbot\System\Generators\Contracts\Generator;

class GeminiGenerator extends Generator
{
    public function generate(): string
    {
        $service = (new GeminiService);

        $response = $service
            ->setHistory([
                'parts' => [
                    [
                        'text' => $this->getPrompt(),
                    ],
                ],
                'role' => 'user',
            ])
            ->generateContent($this->getEntity()->value);

        $response = $response->json('candidates.0.content.parts.0.text');

        return $response ?: '';
    }
}
