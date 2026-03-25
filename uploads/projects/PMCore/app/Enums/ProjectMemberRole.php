<?php

namespace Modules\PMCore\app\Enums;

enum ProjectMemberRole: string
{
    case MANAGER = 'manager';
    case LEAD = 'lead';
    case COORDINATOR = 'coordinator';
    case MEMBER = 'member';
    case VIEWER = 'viewer';
    case CLIENT = 'client';

    public function label(): string
    {
        return match ($this) {
            self::MANAGER => __('Manager'),
            self::LEAD => __('Lead'),
            self::COORDINATOR => __('Coordinator'),
            self::MEMBER => __('Member'),
            self::VIEWER => __('Viewer'),
            self::CLIENT => __('Client'),
        };
    }

    public function permissions(): array
    {
        return match ($this) {
            self::MANAGER => ['view', 'edit', 'delete', 'manage_members', 'manage_tasks', 'manage_budget'],
            self::LEAD => ['view', 'edit', 'manage_tasks', 'assign_tasks', 'log_time'],
            self::COORDINATOR => ['view', 'edit_tasks', 'create_tasks', 'manage_schedule', 'log_time'],
            self::MEMBER => ['view', 'edit_tasks', 'create_tasks', 'log_time'],
            self::VIEWER => ['view'],
            self::CLIENT => ['view', 'comment'],
        };
    }

    public function color(): string
    {
        return match ($this) {
            self::MANAGER => '#007bff',
            self::MEMBER => '#28a745',
            self::VIEWER => '#6c757d',
            self::CLIENT => '#17a2b8',
        };
    }

    public function icon(): string
    {
        return match ($this) {
            self::MANAGER => 'bx bx-crown',
            self::MEMBER => 'bx bx-user',
            self::VIEWER => 'bx bx-low-vision',
            self::CLIENT => 'bx bx-briefcase',
        };
    }

    public static function getDefault(): self
    {
        return self::MEMBER;
    }
}
