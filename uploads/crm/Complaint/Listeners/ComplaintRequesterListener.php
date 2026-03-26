<?php

namespace Modules\Complaint\Listeners;

use Modules\Complaint\Events\ComplaintRequesterEvent;
use Modules\Complaint\Notifications\NewComplaintRequester;
use Illuminate\Support\Facades\Notification;

class ComplaintRequesterListener
{

    /**
     * @param ComplaintRequesterEvent $event
     */

    public function handle(ComplaintRequesterEvent $event)
    {
        if (!is_null($event->notifyUser)) {
            Notification::send($event->notifyUser, new NewComplaintRequester($event->complaint));
        }
    }

}
