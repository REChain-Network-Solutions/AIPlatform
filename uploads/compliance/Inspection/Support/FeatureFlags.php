<?php

namespace Modules\Inspection\Support;

class FeatureFlags
{
    public const TITAN_LINKS = 'inspection.titan_links';

    public static function titanLinksEnabled(): bool
    {
        return (bool) config(self::TITAN_LINKS, true);
    }
}
