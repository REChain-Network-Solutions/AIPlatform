<?php

namespace App\Enums\AiInfluencer;

use App\Enums\Traits\EnumTo;

enum ProductTabEnum: string
{
    use EnumTo;

    case URL = 'url';
    case MANUAL_UPLOAD = 'manual_upload';

    public function label()
    {
        return match ($this) {
            self::URL           => 'Auto URL',
            self::MANUAL_UPLOAD => 'Manual Upload'
        };
    }
}
