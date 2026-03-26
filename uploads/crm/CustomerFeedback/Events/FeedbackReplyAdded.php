<?php

namespace Modules\CustomerFeedback\Events;

use Illuminate\Queue\SerializesModels;
use Illuminate\Broadcasting\PrivateChannel;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Broadcasting\InteractsWithSockets;
use Modules\CustomerFeedback\Entities\FeedbackReply;

class FeedbackReplyAdded
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $reply;

    public function __construct(FeedbackReply $reply)
    {
        $this->reply = $reply;
    }

    public function broadcastOn()
    {
        return new PrivateChannel('feedback-' . $this->reply->company_id);
    }
}
