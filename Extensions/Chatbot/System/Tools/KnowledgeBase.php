<?php

namespace App\Extensions\Chatbot\System\Tools;

use App\Domains\Entity\Enums\EntityEnum;
use App\Extensions\Chatbot\System\Embedders\Contracts\EmbedderInterface;
use App\Extensions\Chatbot\System\Embedders\OpenAIEmbedder;
use App\Extensions\Chatbot\System\Models\Chatbot;
use Illuminate\Support\Collection;
use JsonException;

class KnowledgeBase
{
    public Chatbot $chatbot;

    /**
     * @throws JsonException
     */
    public function call(
        EntityEnum $entityEnum,
        string $query,
        Collection $knowledgeBase,
    ): string {

        $generator = $this
            ->generator($entityEnum)
            ->setInput($query)
            ->generate();

        $queryEmbedding = $generator->embeddings[0]->embedding;

        $results = [];

        foreach ($knowledgeBase as $embedding) {

            if (! $embedding['embedding']) {
                continue;
            }

            foreach ($embedding['embedding'] as $em) {
                $similarity = $this->cosineSimilarity(
                    $em['embedding'],
                    $queryEmbedding
                );

                $results[] = [
                    'content'    => $em['content'],
                    'similarity' => $similarity,
                ];
            }
        }

        usort($results, function ($a, $b) {
            return $b['similarity'] <=> $a['similarity'];
        });

        $results = array_slice($results, 0, 5);
        $texts = array_map(function ($r) {
            return $r['content'];
        }, $results);

        return json_encode($texts, JSON_THROW_ON_ERROR);
    }

    public function generator(EntityEnum $entityEnum): EmbedderInterface
    {
        return app(OpenAIEmbedder::class)->setEntity($entityEnum)->setUser($this->chatbot->user);
    }

    private function cosineSimilarity($vec1, $vec2)
    {
        $dot_product = 0.0;
        $vec1_magnitude = 0.0;
        $vec2_magnitude = 0.0;

        for ($i = 0; $i < count($vec1); $i++) {
            $dot_product += $vec1[$i] * $vec2[$i];
            $vec1_magnitude += $vec1[$i] * $vec1[$i];
            $vec2_magnitude += $vec2[$i] * $vec2[$i];
        }

        $vec1_magnitude = sqrt($vec1_magnitude);
        $vec2_magnitude = sqrt($vec2_magnitude);

        if ($vec1_magnitude == 0.0 || $vec2_magnitude == 0.0) {
            return 0.0; // to handle division by zero
        }

        return $dot_product / ($vec1_magnitude * $vec2_magnitude);
    }

    public function getChatbot(): Chatbot
    {
        return $this->chatbot;
    }

    public function setChatbot(Chatbot $chatbot): self
    {
        $this->chatbot = $chatbot;

        return $this;
    }
}
