<?php

declare(strict_types=1);

namespace App\Services;

use Throwable;

class UpdateChannel
{
    public const KEY = 'update_channel';

    public const STABLE = 'stable';

    public const BETA = 'beta';

    public const STABLE_BASE_V2 = 'https://api.liquid-themes.com/magicai/updater-v2';

    public const BETA_BASE_V2 = 'https://api.liquid-themes.com/magicai/update-test';

    public const STABLE_BASE_V3 = 'https://api.liquid-themes.com/magicai/updater-v3/';

    public const BETA_BASE_V3 = 'https://api.liquid-themes.com/magicai/update-test/';

    public static function channel(): string
    {
        try {
            $stored = setting(self::KEY);
        } catch (Throwable) {
            $stored = null;
        }

        if ($stored === self::BETA || $stored === self::STABLE) {
            return $stored;
        }

        return self::legacyChannel();
    }

    public static function isBeta(): bool
    {
        return self::channel() === self::BETA;
    }

    public static function set(string $channel): string
    {
        $channel = $channel === self::BETA ? self::BETA : self::STABLE;

        setting([self::KEY => $channel])->save();

        return $channel;
    }

    /**
     * Transitional fallback for installations that still carry the pre-security-patch
     * MAGICAI_USER_TYPE flag in .env and never stored a DB value. Once an admin
     * toggles the channel via the UI, the DB value wins permanently and .env is inert.
     */
    protected static function legacyChannel(): string
    {
        return env('MAGICAI_USER_TYPE', '') === 'tester' ? self::BETA : self::STABLE;
    }
}
