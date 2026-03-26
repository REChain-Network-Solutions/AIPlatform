<?php

namespace Modules\Inspection\Providers;

use App\Events\NewCompanyCreatedEvent;

use Modules\Inspection\Entities\Schedule;
use Modules\Inspection\Listeners\CompanyCreatedListener;
use Modules\Inspection\Entities\RecurringSchedule;
use Modules\Inspection\Observers\ScheduleObserver;
use Modules\Inspection\Observers\ScheduleRecurringObserver;
use Illuminate\Foundation\Support\Providers\EventServiceProvider as ServiceProvider;



class EventServiceProvider extends ServiceProvider
{
/**
     * The event listener mappings for the application.
     *
     * @var array
     */
    protected $listen = [

        NewCompanyCreatedEvent::class => [CompanyCreatedListener::class],

    ];



    protected $observers = [
        RecurringSchedule::class => [ScheduleRecurringObserver::class],
        Schedule::class => [ScheduleObserver::class],
    ];

}
