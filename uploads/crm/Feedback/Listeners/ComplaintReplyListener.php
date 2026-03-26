<?php

namespace Modules\Feedback\Listeners;

use App\Models\User;
use Modules\Feedback\Events\FeedbackReplyEvent;
use Modules\Feedback\Entities\FeedbackReply;
use Modules\Feedback\Notifications\NewFeedbackReply;
use Illuminate\Support\Facades\Notification;

class FeedbackReplyListener
{

    /**
     * Handle the event.
     *
     * @param FeedbackReplyEvent $event
     * @return void
     */

    public function handle(FeedbackReplyEvent $event)
    {
        if (!is_null($event->notifyUser)) {
            Notification::send($event->notifyUser, new NewFeedbackReply($event->feedbackReply));
        }
        else {
            Notification::send(User::allAdmins($event->feedbackReply->feedback->company->id), new NewFeedbackReply($event->feedbackReply));
        }
    }

}
