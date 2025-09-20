<?php

namespace App\Enums\AiInfluencer;

use App\Enums\Traits\EnumTo;

enum StepEnum: string
{
    use EnumTo;

    case PRODUCT = 'product';
    case DETAILS = 'details';
    case AVATAR = 'avatar';
    case SCRIPT = 'script';
    case COMPOSITION = 'composition';

    public function label()
    {
        return match ($this) {
            self::PRODUCT     => 'Product',
            self::DETAILS     => 'Details',
            self::SCRIPT      => 'Script',
            self::AVATAR      => 'Avatar',
            self::COMPOSITION => 'Composition',
        };
    }
}
