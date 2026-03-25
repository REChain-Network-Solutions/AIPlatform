<?php

namespace Modules\PMCore\app\Enums;

enum ProjectStatus: string
{
    case PLANNING = 'planning';
    case IN_PROGRESS = 'in_progress';
    case ON_HOLD = 'on_hold';
    case COMPLETED = 'completed';
    case CANCELLED = 'cancelled';

    public function label(): string
    {
        return match ($this) {
            self::PLANNING => __('Planning'),
            self::IN_PROGRESS => __('In Progress'),
            self::ON_HOLD => __('On Hold'),
            self::COMPLETED => __('Completed'),
            self::CANCELLED => __('Cancelled'),
        };
    }

    public function color(): string
    {
        return match ($this) {
            self::PLANNING => 'secondary',
            self::IN_PROGRESS => 'primary',
            self::ON_HOLD => 'warning',
            self::COMPLETED => 'success',
            self::CANCELLED => 'danger',
        };
    }

    public function hexColor(): string
    {
        return match ($this) {
            self::PLANNING => '#6c757d',
            self::IN_PROGRESS => '#007bff',
            self::ON_HOLD => '#ffc107',
            self::COMPLETED => '#28a745',
            self::CANCELLED => '#dc3545',
        };
    }

    public function icon(): string
    {
        return match ($this) {
            self::PLANNING => 'bx bx-edit',
            self::IN_PROGRESS => 'bx bx-play-circle',
            self::ON_HOLD => 'bx bx-pause-circle',
            self::COMPLETED => 'bx bx-check-circle',
            self::CANCELLED => 'bx bx-x-circle',
        };
    }

    public function defaultColor(): string
    {
        return $this->hexColor();
    }

    public static function getDefault(): self
    {
        return self::PLANNING;
    }
}
