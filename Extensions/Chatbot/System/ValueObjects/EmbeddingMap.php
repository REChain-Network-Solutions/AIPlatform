<?php

namespace App\Extensions\Chatbot\System\ValueObjects;

use Illuminate\Contracts\Support\Arrayable;

class EmbeddingMap implements Arrayable
{
    public function __construct(
        public string $content,
        public array $embedding
    ) {}

    public function toArray(): array
    {
        return [
            'content'   => $this->content,
            'embedding' => $this->embedding,
        ];
    }
}
