<?php

namespace App\Extensions\AISocialMedia\System\Services\Contracts;

use App\Extensions\AISocialMedia\System\Models\ScheduledPost;
use Illuminate\Support\Fluent;

abstract class BaseService
{
    private Fluent $platform;

    private ScheduledPost $post;

    abstract public function share($text): void;

    public function getPlatform(): ?Fluent
    {
        return $this->platform;
    }

    public function setPlatform(?Fluent $platform): self
    {
        $this->platform = $platform;

        return $this;
    }

    public function getPost(): ScheduledPost
    {
        return $this->post;
    }

    public function setPost(ScheduledPost $post): self
    {
        $this->post = $post;

        return $this;
    }
}
