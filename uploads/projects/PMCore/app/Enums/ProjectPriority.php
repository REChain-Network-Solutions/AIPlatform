<?php

namespace Modules\PMCore\app\Enums;

enum ProjectPriority: string
{
    case LOW = 'low';
    case MEDIUM = 'medium';
    case HIGH = 'high';
    case URGENT = 'urgent';

    public function label(): string
    {
        return match ($this) {
            self::LOW => __('Low'),
            self::MEDIUM => __('Medium'),
            self::HIGH => __('High'),
            self::URGENT => __('Urgent'),
        };
    }

    public function color(): string
    {
        return match ($this) {
            self::LOW => 'success',
            self::MEDIUM => 'warning',
            self::HIGH => 'info',
            self::URGENT => 'danger',
        };
    }

    public function hexColor(): string
    {
        return match ($this) {
            self::LOW => '#28a745',
            self::MEDIUM => '#ffc107',
            self::HIGH => '#fd7e14',
            self::URGENT => '#dc3545',
        };
    }

    public function icon(): string
    {
        return match ($this) {
            self::LOW => 'bx bx-down-arrow-alt',
            self::MEDIUM => 'bx bx-minus',
            self::HIGH => 'bx bx-up-arrow-alt',
            self::URGENT => 'bx bx-error',
        };
    }

    public static function getDefault(): self
    {
        return self::MEDIUM;
    }
}
