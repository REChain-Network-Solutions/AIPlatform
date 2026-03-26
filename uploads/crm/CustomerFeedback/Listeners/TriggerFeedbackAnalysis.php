<?php

namespace Modules\CustomerFeedback\Listeners;

use Modules\CustomerFeedback\Events\FeedbackTicketCreated;
use Modules\CustomerFeedback\Jobs\AnalyzeFeedbackTicket;

class TriggerFeedbackAnalysis
{
    public function handle(FeedbackTicketCreated $event)
    {
        // Queue AI analysis job
        dispatch(new AnalyzeFeedbackTicket($event->ticket));
    }
}
