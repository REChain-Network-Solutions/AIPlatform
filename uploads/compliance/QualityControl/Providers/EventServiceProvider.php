<?php

namespace Modules\QualityControl\Providers;

use App\Events\NewCompanyCreatedEvent;

use Modules\QualityControl\Entities\Schedule;
use Modules\QualityControl\Listeners\CompanyCreatedListener;
use Modules\QualityControl\Entities\RecurringSchedule;
use Modules\QualityControl\Observers\ScheduleObserver;
use Modules\QualityControl\Observers\ScheduleRecurringObserver;
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
