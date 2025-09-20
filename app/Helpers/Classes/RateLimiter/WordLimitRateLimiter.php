<?php

namespace App\Helpers\Classes\RateLimiter;

use App\Helpers\Classes\Helper;
use Illuminate\Support\Facades\Cache;

class WordLimitRateLimiter
{
    protected string $prefix;

    protected int $maxWordsPerDay;

    public function __construct(
        string $prefix,
        int $maxWordsPerDay = 1500
    ) {
        $this->prefix = $prefix;
        $this->maxWordsPerDay = $maxWordsPerDay;
    }

    public function checkAndRecordInput(string $inputText, ?string $identifier = null): array
    {
        $identifier = $identifier ?? Helper::getRequestIp();
        $wordCount = $this->countWords($inputText);

        $currentUsage = $this->getUsedWords($identifier);
        $remaining = $this->getRemainingWords($identifier);

        if (($currentUsage + $wordCount) > $this->maxWordsPerDay) {
            return [
                'allowed'         => false,
                'word_count'      => $wordCount,
                'used_words'      => $currentUsage,
                'remaining_words' => $remaining,
                'would_exceed_by' => ($currentUsage + $wordCount) - $this->maxWordsPerDay,
                'message'         => "This request contains {$wordCount} words. Your daily limit of {$this->maxWordsPerDay} words would be exceeded by {($currentUsage + $wordCount) - $this->maxWordsPerDay} words.",            ];
        }

        $this->useWords($wordCount, $identifier);

        return [
            'allowed'         => true,
            'word_count'      => $wordCount,
            'used_words'      => $currentUsage + $wordCount,
            'remaining_words' => $this->maxWordsPerDay - ($currentUsage + $wordCount),
            'message'         => "Request accepted. {$wordCount} words have been used.",
        ];
    }

    public function canUseWords(int $wordCount, ?string $identifier = null): bool
    {
        $identifier = $identifier ?? Helper::getRequestIp();
        $currentUsage = $this->getUsedWords($identifier);

        return ($currentUsage + $wordCount) <= $this->maxWordsPerDay;
    }

    public function useWords(int $wordCount, ?string $identifier = null): bool
    {
        $identifier = $identifier ?? Helper::getRequestIp();

        if (! $this->canUseWords($wordCount, $identifier)) {
            return false;
        }

        $cacheKey = $this->buildCacheKey($identifier);
        $currentUsage = Cache::get($cacheKey, 0);
        $newUsage = $currentUsage + $wordCount;

        $expiresAt = now()->endOfDay();
        Cache::put($cacheKey, $newUsage, $expiresAt);

        return true;
    }

    public function processInput(string $inputText, ?string $identifier = null): bool
    {
        $wordCount = $this->countWords($inputText);

        return $this->useWords($wordCount, $identifier);
    }

    public function getUsedWords(?string $identifier = null): int
    {
        $identifier = $identifier ?? Helper::getRequestIp();
        $cacheKey = $this->buildCacheKey($identifier);

        return Cache::get($cacheKey, 0);
    }

    public function getRemainingWords(?string $identifier = null): int
    {
        $identifier = $identifier ?? Helper::getRequestIp();
        $usedWords = $this->getUsedWords($identifier);

        return max(0, $this->maxWordsPerDay - $usedWords);
    }

    public function getUsagePercentage(?string $identifier = null): float
    {
        $identifier = $identifier ?? Helper::getRequestIp();
        $usedWords = $this->getUsedWords($identifier);

        return round(($usedWords / $this->maxWordsPerDay) * 100, 2);
    }

    public function reset(?string $identifier = null): void
    {
        $identifier = $identifier ?? Helper::getRequestIp();
        $cacheKey = $this->buildCacheKey($identifier);
        Cache::forget($cacheKey);
    }

    public function isLimitReached(?string $identifier = null): bool
    {
        $identifier = $identifier ?? Helper::getRequestIp();

        return $this->getUsedWords($identifier) >= $this->maxWordsPerDay;
    }

    protected function buildCacheKey(string $identifier): string
    {
        $date = now()->format('Y-m-d');

        return "{$this->prefix}:words:{$identifier}:{$date}";
    }

    public static function make(
        string $prefix,
        int $maxWordsPerDay = 1500
    ): self {
        return new self($prefix, $maxWordsPerDay);
    }

    public function countWords(string $text): int
    {
        $text = strip_tags($text);

        $text = preg_replace('/\s+/', ' ', trim($text));

        if (empty($text)) {
            return 0;
        }

        return str_word_count($text);
    }
}
