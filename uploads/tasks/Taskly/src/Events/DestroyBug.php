<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class Destroybug
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $bug;

    public function __construct($bug)
    {
        $this->bug = $bug;
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
