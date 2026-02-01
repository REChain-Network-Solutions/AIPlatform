<?php

namespace App\Extensions\Chatbot\System\Generators\Contracts;

use App\Domains\Entity\Enums\EntityEnum;

interface GeneratorInterface
{
    public function setPrompt(string $prompt): static;

    public function getPrompt(): string;

    public function getEntity(): EntityEnum;

    public function setEntity(EntityEnum $entity): static;

    public function generate(): string;
}
