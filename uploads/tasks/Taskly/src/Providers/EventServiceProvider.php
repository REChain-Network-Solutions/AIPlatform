<?php

namespace Workdo\Taskly\Providers;

use App\Events\CompanyMenuEvent;
use App\Events\DefaultData;
use App\Events\GivePermissionToRole;
use Illuminate\Foundation\Support\Providers\EventServiceProvider as Provider;
use Workdo\Taskly\Listeners\CompanyMenuListener;
use Workdo\Taskly\Listeners\DataDefault;
use Workdo\Taskly\Listeners\GiveRoleToPermission;

class EventServiceProvider extends Provider
{
    /**
     * Determine if events and listeners should be automatically discovered.
     *
     * @return bool
     */
    protected $listen = [
        CompanyMenuEvent::class => [
            CompanyMenuListener::class
        ],
        GivePermissionToRole::class => [
            GiveRoleToPermission::class
        ],
        DefaultData::class => [
            DataDefault::class
        ],
    ];

    /**
     * Get the listener directories that should be used to discover events.
     *
     * @return array
     */
    protected function discoverEventsWithin()
    {
        return [
            __DIR__ . '/../Listeners',
        ];
    }
}
