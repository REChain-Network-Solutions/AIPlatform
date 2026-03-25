<?php

namespace Modules\EInvoice\Providers;

use App\Events\NewCompanyCreatedEvent;
use Illuminate\Foundation\Support\Providers\EventServiceProvider as ServiceProvider;
use Modules\EInvoice\Listeners\CompanyCreatedListener;

class EventServiceProvider extends ServiceProvider
{
    protected $listen = [
        NewCompanyCreatedEvent::class => [CompanyCreatedListener::class],
    ];

    protected $observers = [

    ];
}
