<?php

declare(strict_types=1);

namespace App\Extensions\Chatbot\System\Parsers;

use App\Domains\Engine\Enums\EngineEnum;
use App\Extensions\Chatbot\System\Enums\EmbeddingTypeEnum;
use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotEmbedding;
use App\Helpers\Classes\Helper;

class LinkParser
{
    private string $baseUrl;

    private array $links = [];

    private int $maxLinks = 30;

    private array $invalidPaths = ['/cdn-cgi/'];

    private array $contents = [];

    public function insertEmbeddings(Chatbot $chatbot): void
    {
        foreach ($this->contents as $url => $data) {
            ChatbotEmbedding::query()
                ->firstOrCreate([
                    'type'       => EmbeddingTypeEnum::website,
                    'chatbot_id' => $chatbot->getKey(),
                    'url'        => $url,
                    'engine'     => EngineEnum::OPEN_AI->value,
                ], [
                    'title'    => $data['title'],
                    'content'  => $data['content'],
                ]);
        }
    }

    public function crawl(bool $single = false): static
    {
        $this->crawlPage($this->baseUrl, $single);

        return $this;
    }

    private function crawlPage(string $url, bool $single): void
    {
        ini_set('max_execution_time', -1); // 5 minutes

        $html = @file_get_contents($url);
        if ($html === false) {
            return; // Handle the error appropriately
        }

        preg_match('/<title>(.*?)<\/title>/si', $html, $titleMatch);
        $title = $titleMatch[1] ?? 'Untitled'; // Default to 'Untitled' if no title found

        // Store the title and content in an associative array
        $this->contents[$url] = [
            'title'   => $title,
            'content' => $this->stripTagsExceptContent($html),
        ];

        if ($single) {
            return; // No need to continue if crawling a single page
        }

        preg_match_all('/<a\s+(?:[^>]*?\s+)?href="([^"]*)"/', $html, $matches);
        foreach ($matches[1] as $link) {
            $absoluteLink = $this->makeAbsoluteUrl($link);
            if ($this->isValidLink($absoluteLink)) {
                $this->links[] = $absoluteLink;
                if (count($this->links) >= $this->maxLinks) {
                    return;
                }
                $this->crawlPage($absoluteLink, false);
            }
        }
    }

    private function makeAbsoluteUrl(string $url): ?string
    {
        if (str_starts_with($url, 'http') || str_starts_with($url, 'https')) {
            return $url;
        }

        if (str_starts_with($url, '/')) {
            return parse_url($this->baseUrl, PHP_URL_SCHEME) . '://' . parse_url($this->baseUrl, PHP_URL_HOST) . $url;
        }

        return null;
    }

    private function isValidLink(?string $url): bool
    {
        return $url &&
            ! in_array($url, $this->links, true) &&
            $this->isSameDomain($url, $this->baseUrl) &&
            ! $this->hasInvalidPath($url) &&
            ! $this->isImage($url);
    }

    private function isSameDomain(string $url1, string $url2): bool
    {
        return parse_url($url1, PHP_URL_HOST) === parse_url($url2, PHP_URL_HOST);
    }

    private function hasInvalidPath(string $url): bool
    {
        return (bool) array_filter($this->invalidPaths, fn ($invalidPath) => str_contains($url, $invalidPath));
    }

    private function isImage(string $url): bool
    {
        $imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'apng', 'avif', 'svg', 'webp', 'ico', 'tiff'];
        $extension = strtolower(pathinfo($url, PATHINFO_EXTENSION));

        return in_array($extension, $imageExtensions, true);
    }

    private function stripTagsExceptContent(string $html): string
    {
        $patterns = [
            '/<header\b[^>]*>.*?<\/header>/is',
            '/<footer\b[^>]*>.*?<\/footer>/is',
            '/<[^>]+class="[^"]*\bscreen-reader-text\b[^"]*"[^>]*>.*?<\/[^>]+>/is',
            '/<[^>]+class="[^"]*\bscreen-reader-shortcut\b[^"]*"[^>]*>.*?<\/[^>]+>/is',
        ];
        foreach ($patterns as $pattern) {
            $html = preg_replace($pattern, '', $html);
        }

        return Helper::strip_all_tags($html, true);
    }

    public function getContents(): array
    {
        return $this->contents;
    }

    public function getLinks(): array
    {
        return $this->links;
    }

    public function getBaseUrl(): string
    {
        return $this->baseUrl;
    }

    public function setBaseUrl(string $baseUrl): self
    {
        $this->baseUrl = $baseUrl;

        return $this;
    }
}
