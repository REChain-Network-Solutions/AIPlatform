<?php

declare(strict_types=1);

namespace App\Extensions\Announcement\System\Enum;

use App\Enums\Contracts\WithStringBackedEnum;
use App\Enums\Traits\EnumTo;
use App\Enums\Traits\StringBackedEnumTrait;

enum AnnouncementType: string implements WithStringBackedEnum
{
    use EnumTo;
    use StringBackedEnumTrait;

    case NEW = 'new';
    case MAINTENANCE = 'maintenance';
    case UPDATE = 'update';
    case INFO = 'info';

    public function label(): string
    {
        return match ($this) {
            self::NEW 			      => __('New'),
            self::MAINTENANCE 	=> __('Maintenance'),
            self::UPDATE 		    => __('Update'),
            self::INFO 			     => __('Info')
        };
    }

    public function image(): string
    {
        return match ($this) {
            self::NEW 			         => 'speakerphone',
            self::MAINTENANCE     => 'tool',
            self::UPDATE 		       => 'refresh',
            self::INFO 			        => 'info-hexagon'
        };
    }

    public function color(): string
    {
        return match ($this) {
            self::NEW 			         => '#9A34CD',
            self::MAINTENANCE     => '#42444A',
            self::UPDATE 		       => '#16A34A',
            self::INFO 			        => '#49AEDE'
        };
    }
}
