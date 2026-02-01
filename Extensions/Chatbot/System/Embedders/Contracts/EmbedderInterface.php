<?php

namespace App\Extensions\Chatbot\System\Embedders\Contracts;

use App\Domains\Entity\Enums\EntityEnum;

interface EmbedderInterface
{
    public function getInput(): string|array;

    public function setInput(string|array $input): static;

    public function getEntity(): EntityEnum;

    public function setEntity(EntityEnum $entity): static;

    public function generate();
}
