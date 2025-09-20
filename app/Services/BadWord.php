<?php

namespace App\Services;

use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Auth;

class BadWord
{
    protected bool $error = false;

    protected array $badWords = [];

    protected Collection $errorWords;

    /**
     * BadWord constructor.
     */
    public function __construct(
        public string $text,
    ) {
        $this->errorWords = collect();

        $this->check();
    }

    public function check(): void
    {
        foreach ($this->getBadWords() as $badWord) {
            if (str_contains(strtolower($this->text), $badWord)) {
                $this->error = true;
                $this->errorWords->push($badWord);
            }
        }
    }

    public function hasError(): bool
    {
        return $this->error;
    }

    public function getErrors(): Collection
    {
        return $this->errorWords;
    }

    protected function getBadWords(): array
    {
        return Auth::user()->badWord->getWordsAsArray();
    }
}
