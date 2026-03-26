<?php

namespace Modules\Feedback\Listeners;

use Modules\Feedback\Events\FeedbackRequesterEvent;
use Modules\Feedback\Notifications\NewFeedbackRequester;
use Illuminate\Support\Facades\Notification;

class FeedbackRequesterListener
{

    /**
     * @param FeedbackRequesterEvent $event
     */

    public function handle(FeedbackRequesterEvent $event)
    {
        if (!is_null($event->notifyUser)) {
            Notification::send($event->notifyUser, new NewFeedbackRequester($event->feedback));
        }
    }

}
