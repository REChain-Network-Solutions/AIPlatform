<?php

namespace Modules\CustomerFeedback\Listeners;

use Modules\CustomerFeedback\Events\FeedbackTicketCreated;
use Modules\CustomerFeedback\Notifications\NewFeedbackTicket;

class SendFeedbackNotification
{
    public function handle(FeedbackTicketCreated $event)
    {
        // Notify assigned agent if ticket is assigned
        if ($event->ticket->agent_id) {
            $event->ticket->agent->notify(new NewFeedbackTicket($event->ticket));
        }

        // Notify group members if ticket is assigned to a group
        if ($event->ticket->group_id && $event->ticket->group) {
            foreach ($event->ticket->group->enabledAgents as $agent) {
                $agent->notify(new NewFeedbackTicket($event->ticket));
            }
        }
    }
}
