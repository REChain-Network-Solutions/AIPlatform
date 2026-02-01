<?php

namespace App\Extensions\Chatbot\System\Enums;

use App\Enums\Traits\EnumTo;

enum StepEnum: string
{
    use EnumTo;

    case configure = 'configure';
    case customize = 'customize';
    case train = 'train';
    case embed = 'embed';
    case channel = 'channel';

    public static function isInValid(string $step): bool
    {
        return ! self::isValid($step);
    }

    public static function isValid(string $step): bool
    {
        return in_array($step, self::toArray(), true);
    }
}
