<?php

namespace Modules\CustomerFeedback\Events;

use Illuminate\Queue\SerializesModels;
use Illuminate\Broadcasting\PrivateChannel;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Broadcasting\InteractsWithSockets;
use Modules\CustomerFeedback\Entities\NpsSurvey;

class NpsSurveyCreated
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public $survey;

    public function __construct(NpsSurvey $survey)
    {
        $this->survey = $survey;
    }

    public function broadcastOn()
    {
        return new PrivateChannel('feedback-' . $this->survey->company_id);
    }
}
