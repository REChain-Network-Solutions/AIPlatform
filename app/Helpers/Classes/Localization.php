<?php

namespace App\Helpers\Classes;

class Localization
{
    public static function countyCodes()
    {
        return config('localization.countries');
    }

    public static function getSupportedLocales(): array
    {
        return config('localization.supportedLocales');
    }

    public static function setLocale(string $value): void
    {
        session()->put('app_locale', $value);
        session()->save();
    }

    public static function getCurrentLocaleDirection()
    {

        if (! empty(self::getSupportedLocales()[self::getLocale()]['dir'])) {
            return self::getSupportedLocales()[self::getLocale()]['dir'];
        }

        switch (self::getCurrentLocaleScript()) {
            // Other (historic) RTL scripts exist, but this list contains the only ones in current use.
            case 'Arab':
            case 'Hebr':
            case 'Mong':
            case 'Tfng':
            case 'Thaa':
                return 'rtl';
            default:
                return 'ltr';
        }
    }

    public static function getCurrentLocaleScript()
    {
        return self::getSupportedLocales()[self::getLocale()]['script'];
    }

    public static function getLocale(): ?string
    {
        return session('app_locale') ?: app()->getLocale();
    }
}
