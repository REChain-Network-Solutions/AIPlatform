<?php

namespace Modules\PMCore\app\Enums;

enum TimesheetStatus: string
{
    case DRAFT = 'draft';
    case SUBMITTED = 'submitted';
    case APPROVED = 'approved';
    case REJECTED = 'rejected';
    case INVOICED = 'invoiced';

    public function label(): string
    {
        return match ($this) {
            self::DRAFT => 'Draft',
            self::SUBMITTED => 'Submitted',
            self::APPROVED => 'Approved',
            self::REJECTED => 'Rejected',
            self::INVOICED => 'Invoiced',
        };
    }

    public function color(): string
    {
        return match ($this) {
            self::DRAFT => 'secondary',
            self::SUBMITTED => 'warning',
            self::APPROVED => 'success',
            self::REJECTED => 'danger',
            self::INVOICED => 'info',
        };
    }

    public function icon(): string
    {
        return match ($this) {
            self::DRAFT => 'bx-edit',
            self::SUBMITTED => 'bx-time',
            self::APPROVED => 'bx-check',
            self::REJECTED => 'bx-x',
            self::INVOICED => 'bx-receipt',
        };
    }

    public static function getOptions(): array
    {
        return [
            self::DRAFT->value => self::DRAFT->label(),
            self::SUBMITTED->value => self::SUBMITTED->label(),
            self::APPROVED->value => self::APPROVED->label(),
            self::REJECTED->value => self::REJECTED->label(),
            self::INVOICED->value => self::INVOICED->label(),
        ];
    }
}
