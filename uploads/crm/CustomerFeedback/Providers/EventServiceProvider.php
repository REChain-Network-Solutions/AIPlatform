<?php

namespace Modules\CustomerFeedback\Providers;

use Illuminate\Foundation\Support\Providers\EventServiceProvider as ServiceProvider;
use Modules\CustomerFeedback\Events\FeedbackTicketCreated;
use Modules\CustomerFeedback\Events\FeedbackTicketUpdated;
use Modules\CustomerFeedback\Events\FeedbackReplyAdded;
use Modules\CustomerFeedback\Events\NpsSurveyCreated;
use Modules\CustomerFeedback\Listeners\SendFeedbackNotification;
use Modules\CustomerFeedback\Listeners\TriggerFeedbackAnalysis;

class EventServiceProvider extends ServiceProvider
{
    protected $listen = [
        FeedbackTicketCreated::class => [
            SendFeedbackNotification::class,
            TriggerFeedbackAnalysis::class,
        ],
        FeedbackTicketUpdated::class => [
            SendFeedbackNotification::class,
        ],
        FeedbackReplyAdded::class => [
            SendFeedbackNotification::class,
        ],
        NpsSurveyCreated::class => [
            // Add listeners for NPS surveys if needed
        ],
    ];

    public function boot()
    {
        parent::boot();
    }
}
