<?php

namespace App\Enums\AiInfluencer;

use App\Enums\Traits\EnumTo;

enum VideoStatusEnum: string
{
    use EnumTo;

    case IN_PROGRESS = 'in_progress';
    case FAILED = 'failed';
    case COMPLETED = 'completed';

    public function label()
    {
        return match ($this) {
            self::IN_PROGRESS => 'In Progress',
            self::FAILED      => 'Failed',
            self::COMPLETED   => 'Completed'
        };
    }
}
