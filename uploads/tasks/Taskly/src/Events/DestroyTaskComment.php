<?php

namespace Workdo\Taskly\Events;

use Illuminate\Queue\SerializesModels;

class DestroyTaskComment
{
    use SerializesModels;

    /**
     * Create a new event instance.
     *
     * @return void
     */
    public $comment;

    public function __construct($comment)
    {
        $this->comment = $comment;
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
