<?php

namespace Modules\Feedback\Events;

use Modules\Feedback\Entities\Feedback;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class FeedbackEvent
{

    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $feedback;
    public $notificationName;

    public function __construct(Feedback $feedback, $notificationName)
    {
        $this->feedback = $feedback;
        $this->notificationName = $notificationName;
    }

}
