<?php

namespace Modules\Feedback\Events;

use Modules\Feedback\Entities\FeedbackReply;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class FeedbackReplyEvent
{

    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $feedbackReply;
    public $notifyUser;

    public function __construct(FeedbackReply $feedbackReply, $notifyUser)
    {
        $this->feedbackReply = $feedbackReply;
        $this->notifyUser = $notifyUser;
    }

}
