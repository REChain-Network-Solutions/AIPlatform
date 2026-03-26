<?php

namespace Modules\Complaint\Listeners;

use App\Models\User;
use Modules\Complaint\Events\ComplaintReplyEvent;
use Modules\Complaint\Entities\ComplaintReply;
use Modules\Complaint\Notifications\NewComplaintReply;
use Illuminate\Support\Facades\Notification;

class ComplaintReplyListener
{

    /**
     * Handle the event.
     *
     * @param ComplaintReplyEvent $event
     * @return void
     */

    public function handle(ComplaintReplyEvent $event)
    {
        if (!is_null($event->notifyUser)) {
            Notification::send($event->notifyUser, new NewComplaintReply($event->complaintReply));
        }
        else {
            Notification::send(User::allAdmins($event->complaintReply->complaint->company->id), new NewComplaintReply($event->complaintReply));
        }
    }

}
