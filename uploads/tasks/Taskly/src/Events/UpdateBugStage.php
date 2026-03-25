<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class UpdateBugStage
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $request;
    public $bug;

    public function __construct($request,$bug)
    {
        $this->request  = $request;
        $this->bug      = $bug;
    }

    /**
     * Get the channels the event should be broadcast on.
     *
     * @return array
     */
    public function broadcastOn()
    {
        return [];
    }
}
