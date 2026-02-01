<?php

namespace App\Extensions\Chatbot\System\Parsers;

use Smalot\PdfParser\Parser;

class PdfParser
{
    public string $text;

    public string $pdfPath;

    public function __construct(
        public Parser $parser
    ) {}

    public function parse(): string
    {
        $content = $this->parser
            ->parseFile($this->getPath())
            ->getText();

        $this->setText($content);

        return $this->getText();
    }

    public function setPath(string $pdfPath): PdfParser
    {
        $this->pdfPath = $pdfPath;

        return $this;
    }

    public function getPath(): string
    {
        return $this->pdfPath;
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
