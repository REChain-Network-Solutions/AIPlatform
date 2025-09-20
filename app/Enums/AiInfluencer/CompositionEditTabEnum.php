<?php

namespace App\Enums\AiInfluencer;

use App\Enums\Traits\EnumTo;

enum CompositionEditTabEnum: string
{
    use EnumTo;

    case AVATAR = 'avatar';
    case VOICE = 'voice';
    case MUSIC = 'music';
    case CAPTIONS = 'captions';

    public function label(): string
    {
        return match ($this) {
            self::AVATAR   => 'Avatar',
            self::VOICE    => 'Voice',
            self::MUSIC    => 'Music',
            self::CAPTIONS => 'Captions'
        };
    }

    public function svg(): string
    {
        return match ($this) {
            self::AVATAR   => 'tabler-user-circle',
            self::VOICE    => 'tabler-microphone-2',
            self::MUSIC    => 'tabler-music',
            self::CAPTIONS => 'tabler-text-caption'
        };
    }
}
