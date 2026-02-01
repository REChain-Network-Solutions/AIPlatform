<?php

namespace App\Extensions\Chatbot\System\ValueObjects;

use Illuminate\Contracts\Support\Arrayable;

class Embedding implements Arrayable
{
    public readonly ?array $value;

    public function __construct(EmbeddingMap ...$maps)
    {
        $value = [];

        foreach ($maps as $map) {
            $value[] = [
                'content'   => $map->content,
                'embedding' => $map->embedding,
            ];
        }

        $this->value = $value;
    }

    public function toArray(): ?array
    {
        return $this->value;
    }
}
