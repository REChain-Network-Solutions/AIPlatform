<?php

namespace App\Extensions\Chatbot\System\Parsers;

use Illuminate\Validation\ValidationException;

class TextParser
{
    public string $text;

    public string $path;

    public function parse(): string
    {
        $content = file($this->getPath());

        if (empty($content)) {
            throw ValidationException::withMessages([
                'file' => trans('File format is not correct.'),
            ]);
        }

        if (is_array($content)) {
            $content = implode('', $content);
        }

        $this->setText($content);

        return $this->getText();
    }

    public function setPath(string $path): TextParser
    {
        $this->path = $path;

        return $this;
    }

    public function getPath(): string
    {
        return $this->path;
    }

    public function getText(): string
    {
        return $this->text;
    }

    public function setText(string $text): void
    {
        $this->text = $text;
    }
}
