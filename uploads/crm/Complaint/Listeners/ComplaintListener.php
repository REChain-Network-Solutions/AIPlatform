<?php

namespace Modules\Complaint\Listeners;

use Modules\Complaint\Events\ComplaintEvent;
use Modules\Complaint\Notifications\NewComplaint;
use Modules\Complaint\Notifications\ComplaintAgent;
use App\Models\User;
use Illuminate\Support\Facades\Notification;

class ComplaintListener
{

    /**
     * Handle the event.
     *
     * @param ComplaintEvent $event
     * @return void
     */

    public function handle(ComplaintEvent $event)
    {
        if ($event->notificationName == 'NewComplaint') {
            Notification::send(User::allAdmins(), new NewComplaint($event->complaint));
        }
        elseif ($event->notificationName == 'ComplaintAgent') {
            Notification::send($event->complaint->agent, new ComplaintAgent($event->complaint));
        }
    }

}
