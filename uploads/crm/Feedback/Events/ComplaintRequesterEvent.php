<?php

namespace Modules\Feedback\Events;

use Modules\Feedback\Entities\Feedback;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class FeedbackRequesterEvent
{

    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $feedback;
    public $notifyUser;

    public function __construct(Feedback $feedback, $notifyUser)
    {
        $this->feedback = $feedback;
        $this->notifyUser = $notifyUser;
    }

}
