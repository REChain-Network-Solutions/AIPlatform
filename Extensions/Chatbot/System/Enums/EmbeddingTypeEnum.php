<?php

namespace App\Extensions\Chatbot\System\Enums;

use App\Enums\Traits\EnumTo;

enum EmbeddingTypeEnum: string
{
    use EnumTo;

    case website = 'website';
    case file = 'file';
    case text = 'text';
    case qa = 'qa';

    public static function isInValid(string $step): bool
    {
        return ! self::isValid($step);
    }

    public static function isValid(string $step): bool
    {
        return in_array($step, self::toArray(), true);
    }
}
