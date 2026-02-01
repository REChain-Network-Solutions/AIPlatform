<?php

namespace App\Extensions\Chatbot\System\Embedders\Contracts;

use App\Domains\Entity\Enums\EntityEnum;
use App\Models\User;

abstract class Embedder implements EmbedderInterface
{
    public EntityEnum $entity;

    public string|array $input;

    public User $user;

    public function getEntity(): EntityEnum
    {
        return $this->entity;
    }

    public function setEntity(EntityEnum $entity): static
    {
        $this->entity = $entity;

        return $this;
    }

    public function setInput(array|string $input): static
    {
        $this->input = $input;

        return $this;
    }

    public function getInput(): array|string
    {
        return $this->input;
    }

    public function getUser(): User
    {
        return $this->user;
    }

    public function setUser(User $user): static
    {
        $this->user = $user;

        return $this;
    }
}
