<?php

namespace App\Enums\AiInfluencer;

use App\Enums\Traits\EnumTo;

enum ScriptTabEnum: string
{
    use EnumTo;

    case AUTO_GENERATED_SCRIPT = 'auto_generated_script';
    case CUSTOM_SCRIPT = 'custom_script';

    public function label()
    {
        return match ($this) {
            self::AUTO_GENERATED_SCRIPT => 'Auto Generated Script',
            self::CUSTOM_SCRIPT         => 'Custom Script'
        };
    }
}
