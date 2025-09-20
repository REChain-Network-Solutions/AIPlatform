<?php

declare(strict_types=1);

namespace App\Enums;

use App\Enums\Contracts\WithStringBackedEnum;
use App\Enums\Traits\EnumTo;
use App\Enums\Traits\StringBackedEnumTrait;

enum AccessType: string implements WithStringBackedEnum
{
    use EnumTo;
    use StringBackedEnumTrait;

    // Sorted from best to lowest
    case VIP = 'vip';
    case ENTERPRISE = 'enterprise';
    case PRO = 'pro';
    case PREMIUM = 'premium';
    case REGULAR = 'regular';

    /**
     * Get a human-readable label for the enum case.
     */
    public function label(): string
    {
        return match ($this) {
            self::VIP        => __('VIP'),
            self::ENTERPRISE => __('Enterprise'),
            self::PRO        => __('Pro'),
            self::PREMIUM    => __('Premium'),
            self::REGULAR    => __('Regular'),
        };
    }

    /**
     * Get a color associated with the access type in HSL format.
     * Tone is kept consistent (high lightness and saturation) but hue differs.
     */
    public function color(): string
    {
        return match ($this) {
            self::VIP        => '278, 87%, 94%',
            self::ENTERPRISE => '210, 87%, 94%',
            self::PRO        => '145, 87%, 94%',
            self::PREMIUM    => '34, 87%, 94%',
            self::REGULAR    => '0, 87%, 94%',
        };
    }

    /**
     * Get all values of the enum.
     *
     * @return array<string>
     */
    public static function getValues(): array
    {
        return array_map(static fn ($value) => $value->value, self::cases());
    }
}
