<?php

namespace Modules\PMCore\app\Enums;

enum ProjectType: string
{
    case INTERNAL = 'internal';
    case CLIENT = 'client';
    case MAINTENANCE = 'maintenance';
    case DEVELOPMENT = 'development';

    public function label(): string
    {
        return match ($this) {
            self::INTERNAL => __('Internal'),
            self::CLIENT => __('Client Work'),
            self::MAINTENANCE => __('Maintenance'),
            self::DEVELOPMENT => __('Development'),
        };
    }

    public function color(): string
    {
        return match ($this) {
            self::INTERNAL => '#6f42c1',
            self::CLIENT => '#007bff',
            self::MAINTENANCE => '#ffc107',
            self::DEVELOPMENT => '#28a745',
        };
    }

    public function icon(): string
    {
        return match ($this) {
            self::INTERNAL => 'bx bx-building',
            self::CLIENT => 'bx bx-user',
            self::MAINTENANCE => 'bx bx-wrench',
            self::DEVELOPMENT => 'bx bx-code-alt',
        };
    }

    public static function getDefault(): self
    {
        return self::CLIENT;
    }
}
