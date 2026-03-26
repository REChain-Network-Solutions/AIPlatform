<?php

namespace Modules\Feedback\Listeners;

use Modules\Feedback\Events\FeedbackEvent;
use Modules\Feedback\Notifications\NewFeedback;
use Modules\Feedback\Notifications\FeedbackAgent;
use App\Models\User;
use Illuminate\Support\Facades\Notification;

class FeedbackListener
{

    /**
     * Handle the event.
     *
     * @param FeedbackEvent $event
     * @return void
     */

    public function handle(FeedbackEvent $event)
    {
        if ($event->notificationName == 'NewFeedback') {
            Notification::send(User::allAdmins(), new NewFeedback($event->feedback));
        }
        elseif ($event->notificationName == 'FeedbackAgent') {
            Notification::send($event->feedback->agent, new FeedbackAgent($event->feedback));
        }
    }

}
